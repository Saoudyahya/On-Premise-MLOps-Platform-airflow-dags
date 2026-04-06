from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
from mlflow import MlflowClient
import os
import time
import logging
import hashlib

logger = logging.getLogger(__name__)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000")
ADMIN_USER = "admin"
ADMIN_PASS = "mlops-admin-2024"

NODE_IP = os.environ.get("NODE_IP", "192.168.0.64")


def validate_and_promote(researcher_id, model_name, version, threshold, **ctx):
    import mlflow

    logger.info(f"=== VALIDATE | researcher={researcher_id} model={model_name} v={version} ===")

    os.environ["MLFLOW_TRACKING_URI"]      = MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = ADMIN_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = ADMIN_PASS

    mlflow.set_tracking_uri(MLFLOW_URI)
    admin_client = MlflowClient(tracking_uri=MLFLOW_URI)

    if version == "latest":
        versions = admin_client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        mv = versions[0]
    else:
        mv = admin_client.get_model_version(model_name, version)

    logger.info(f"Found model version: {mv.version} run_id={mv.run_id}")

    run    = admin_client.get_run(mv.run_id)
    acc    = float(run.data.metrics.get("accuracy", 0))
    thresh = float(threshold)

    logger.info(f"accuracy={acc}, threshold={thresh}")
    if acc < thresh:
        raise ValueError(f"accuracy {acc} is below threshold {thresh}")

    admin_client.set_registered_model_alias(model_name, "production", mv.version)
    logger.info(f"✓ Alias 'production' → {model_name} v{mv.version}")

    ctx["ti"].xcom_push(key="model_version", value=mv.version)
    logger.info("=== VALIDATE DONE ===")


def _k8s_clients():
    from kubernetes import client as k8s, config as k8s_config
    k8s_config.load_incluster_config()
    return (
        k8s.CoreV1Api(),
        k8s.AppsV1Api(),
        k8s.AutoscalingV2Api(),
    )


def launch_and_wait_healthy(researcher_id, model_name,
                             replicas=1,
                             hpa_enabled=False,
                             hpa_min_replicas=1,
                             hpa_max_replicas=5,
                             hpa_cpu_target=70,
                             **ctx):
    from kubernetes import client as k8s

    # Normalise types — Airflow Jinja renders everything as strings
    replicas         = int(replicas)
    hpa_enabled      = str(hpa_enabled).lower() in ("true", "1", "yes")
    hpa_min_replicas = int(hpa_min_replicas)
    hpa_max_replicas = int(hpa_max_replicas)
    hpa_cpu_target   = int(hpa_cpu_target)

    logger.info(
        f"=== DEPLOY SERVING DEPLOYMENT | model={model_name} researcher={researcher_id} "
        f"replicas={replicas} hpa={hpa_enabled} "
        f"hpa_range=[{hpa_min_replicas},{hpa_max_replicas}] cpu_target={hpa_cpu_target}% ==="
    )

    core_v1, apps_v1, autoscaling_v2 = _k8s_clients()

    namespace   = "mlops-serving"
    model_slug  = model_name.replace("_", "-").lower()
    rid_slug    = researcher_id.replace("_", "-").lower()
    deploy_name = f"serve-{rid_slug}-{model_slug}"
    svc_name    = f"{deploy_name}-svc"
    hpa_name    = f"{deploy_name}-hpa"

    # NodePort assignments
    node_port = 30000 + (int(hashlib.md5(model_name.encode()).hexdigest(), 16) % 2767)
    metrics_node_port = 30000 + (int(hashlib.md5((model_name + "_metrics").encode()).hexdigest(), 16) % 2767)
    if metrics_node_port == node_port:
        metrics_node_port = node_port + 1 if node_port < 32767 else node_port - 1

    serving_url = f"http://{NODE_IP}:{node_port}/invocations"
    logger.info(f"Inference NodePort : {node_port}  → {serving_url}")
    logger.info(f"Metrics  NodePort  : {metrics_node_port} → http://{NODE_IP}:{metrics_node_port}/metrics")

    # ── 1. Delete existing HPA first (must precede Deployment deletion) ───────
    try:
        autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(hpa_name, namespace)
        logger.info(f"Deleted existing HPA {hpa_name}")
    except k8s.exceptions.ApiException as e:
        if e.status != 404:
            raise

    # ── 2. Delete existing Deployment (Foreground — waits for pods to die) ────
    try:
        apps_v1.delete_namespaced_deployment(
            deploy_name, namespace,
            body=k8s.V1DeleteOptions(propagation_policy="Foreground"),
        )
        logger.info(f"Deleted existing Deployment {deploy_name}, waiting for termination…")
        for _ in range(30):
            time.sleep(3)
            try:
                apps_v1.read_namespaced_deployment(deploy_name, namespace)
            except k8s.exceptions.ApiException as e:
                if e.status == 404:
                    logger.info("Deployment terminated.")
                    break
    except k8s.exceptions.ApiException as e:
        if e.status != 404:
            raise

    # ── 3. Shared labels ──────────────────────────────────────────────────────
    labels = {
        "app":               deploy_name,
        "model":             f"{rid_slug}-{model_slug}",
        "researcher":        rid_slug,
        "prometheus-scrape": "true",
    }
    pod_annotations = {
        "prometheus.io/scrape": "true",
        "prometheus.io/port":   "8082",
        "prometheus.io/path":   "/metrics",
    }

    # ── 4. Container spec ─────────────────────────────────────────────────────
    container = k8s.V1Container(
        name="mlflow-serve",
        image="ghcr.io/mlflow/mlflow:v3.10.1",
        command=["sh", "-c"],
        args=[
            "pip install mlflow boto3 scikit-learn mlserver mlserver-mlflow 'uvloop<0.22' --quiet && "
            "mlflow models serve "
            f"--model-uri 'models:/{model_name}@production' "
            "--host 0.0.0.0 --port 8080 "
            "--enable-mlserver "
            "--no-conda"
        ],
        ports=[
            k8s.V1ContainerPort(container_port=8080, name="inference"),
            k8s.V1ContainerPort(container_port=8081, name="grpc"),
            k8s.V1ContainerPort(container_port=8082, name="metrics"),
        ],
        env=[
            k8s.V1EnvVar(name="MLFLOW_TRACKING_URI",       value=MLFLOW_URI),
            k8s.V1EnvVar(name="MLFLOW_TRACKING_USERNAME",  value=ADMIN_USER),
            k8s.V1EnvVar(name="MLFLOW_TRACKING_PASSWORD",  value=ADMIN_PASS),
            k8s.V1EnvVar(name="MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", value="true"),
            k8s.V1EnvVar(name="MLFLOW_ARTIFACT_URI",      value=MLFLOW_URI),
            k8s.V1EnvVar(name="AWS_ENDPOINT_URL",          value="http://minio-svc.mlops-minio.svc.cluster.local:9000"),
            k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID",         value="minio-admin"),
            k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY",     value="minio-admin"),
            k8s.V1EnvVar(name="MLSERVER_HTTP_PORT",        value="8080"),
            k8s.V1EnvVar(name="MLSERVER_GRPC_PORT",        value="8081"),
            k8s.V1EnvVar(name="MLSERVER_METRICS_PORT",     value="8082"),
            k8s.V1EnvVar(name="MLSERVER_MODEL_NAME",       value=model_name),
            k8s.V1EnvVar(name="MLSERVER_PARALLEL_WORKERS", value="0"),
        ],
        readiness_probe=k8s.V1Probe(
            http_get=k8s.V1HTTPGetAction(path="/v2/health/ready", port=8080),
            initial_delay_seconds=10,
            period_seconds=10,
            failure_threshold=12,
        ),
        liveness_probe=k8s.V1Probe(
            http_get=k8s.V1HTTPGetAction(path="/v2/health/live", port=8080),
            initial_delay_seconds=10,
            period_seconds=15,
            failure_threshold=6,
        ),
        startup_probe=k8s.V1Probe(
            http_get=k8s.V1HTTPGetAction(path="/v2/health/live", port=8080),
            failure_threshold=60,
            period_seconds=10,
        ),
        resources=k8s.V1ResourceRequirements(
            requests={"memory": "512Mi", "cpu": "250m"},
            limits={"memory": "2Gi", "cpu": "1000m"},
        ),
    )

    # ── 5. Create Deployment ──────────────────────────────────────────────────
    deployment = k8s.V1Deployment(
        metadata=k8s.V1ObjectMeta(
            name=deploy_name,
            namespace=namespace,
            labels=labels,
            annotations=pod_annotations,
        ),
        spec=k8s.V1DeploymentSpec(
            replicas=replicas,
            selector=k8s.V1LabelSelector(match_labels={"app": deploy_name}),
            strategy=k8s.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=k8s.V1RollingUpdateDeployment(
                    max_surge=1,
                    max_unavailable=0,
                ),
            ),
            template=k8s.V1PodTemplateSpec(
                metadata=k8s.V1ObjectMeta(
                    labels=labels,
                    annotations=pod_annotations,
                ),
                spec=k8s.V1PodSpec(
                    restart_policy="Always",
                    containers=[container],
                ),
            ),
        ),
    )

    apps_v1.create_namespaced_deployment(namespace, deployment)
    logger.info(f"✓ Deployment {deploy_name} created (replicas={replicas})")

    # ── 6. Create / recreate Service ─────────────────────────────────────────
    service = k8s.V1Service(
        metadata=k8s.V1ObjectMeta(name=svc_name, namespace=namespace, labels=labels),
        spec=k8s.V1ServiceSpec(
            type="NodePort",
            selector={"app": deploy_name},
            ports=[
                k8s.V1ServicePort(name="inference", port=8080, target_port=8080, node_port=node_port),
                k8s.V1ServicePort(name="metrics",   port=8082, target_port=8082, node_port=metrics_node_port),
            ],
        ),
    )

    try:
        core_v1.read_namespaced_service(svc_name, namespace)
        core_v1.delete_namespaced_service(svc_name, namespace)
        logger.info(f"Deleted old Service {svc_name}")
        time.sleep(1)
    except k8s.exceptions.ApiException as e:
        if e.status != 404:
            raise

    core_v1.create_namespaced_service(namespace, service)
    logger.info(f"✓ Service {svc_name} created  inference={node_port}  metrics={metrics_node_port}")

    # ── 7. Create HPA if requested ────────────────────────────────────────────
    if hpa_enabled:
        # Enforce: min_replicas must be <= initial replicas (or HPA will fight deployment)
        effective_min = min(hpa_min_replicas, replicas)

        hpa_metrics = [
            k8s.V2MetricSpec(
                type="Resource",
                resource=k8s.V2ResourceMetricSource(
                    name="cpu",
                    target=k8s.V2MetricTarget(
                        type="Utilization",
                        average_utilization=hpa_cpu_target,
                    ),
                ),
            )
        ]

        hpa = k8s.V2HorizontalPodAutoscaler(
            metadata=k8s.V1ObjectMeta(name=hpa_name, namespace=namespace, labels=labels),
            spec=k8s.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=k8s.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deploy_name,
                ),
                min_replicas=effective_min,
                max_replicas=hpa_max_replicas,
                metrics=hpa_metrics,
                behavior=k8s.V2HorizontalPodAutoscalerBehavior(
                    scale_up=k8s.V2HPAScalingRules(
                        stabilization_window_seconds=60,
                        policies=[
                            k8s.V2HPAScalingPolicy(type="Pods", value=2, period_seconds=60),
                        ],
                    ),
                    scale_down=k8s.V2HPAScalingRules(
                        stabilization_window_seconds=300,   # 5 min cool-down on scale-down
                        policies=[
                            k8s.V2HPAScalingPolicy(type="Pods", value=1, period_seconds=120),
                        ],
                    ),
                ),
            ),
        )
        autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(namespace, hpa)
        logger.info(
            f"✓ HPA {hpa_name} created  "
            f"cpu_target={hpa_cpu_target}%  "
            f"replicas=[{effective_min},{hpa_max_replicas}]"
        )

    # ── 8. Push XCom ─────────────────────────────────────────────────────────
    ctx["ti"].xcom_push(key="serving_url",       value=serving_url)
    ctx["ti"].xcom_push(key="metrics_url",       value=f"http://{NODE_IP}:{metrics_node_port}/metrics")
    ctx["ti"].xcom_push(key="metrics_node_port", value=metrics_node_port)
    ctx["ti"].xcom_push(key="node_port",         value=node_port)
    # NOTE: pod_name now holds the Deployment name.
    # Prometheus queries use the `app=<deploy_name>` label, which aggregates all replicas.
    ctx["ti"].xcom_push(key="pod_name",          value=deploy_name)
    ctx["ti"].xcom_push(key="svc_name",          value=svc_name)
    ctx["ti"].xcom_push(key="node_ip",           value=NODE_IP)
    ctx["ti"].xcom_push(key="replicas",          value=replicas)
    ctx["ti"].xcom_push(key="hpa_enabled",       value=hpa_enabled)
    ctx["ti"].xcom_push(key="hpa_min_replicas",  value=hpa_min_replicas if hpa_enabled else None)
    ctx["ti"].xcom_push(key="hpa_max_replicas",  value=hpa_max_replicas if hpa_enabled else None)
    ctx["ti"].xcom_push(key="hpa_cpu_target",    value=hpa_cpu_target   if hpa_enabled else None)

    # ── 9. Poll until Deployment has ≥1 ready replica ────────────────────────
    logger.info("Waiting for Deployment to have Ready replicas (max 10 min)…")
    for attempt in range(120):
        time.sleep(5)
        dep     = apps_v1.read_namespaced_deployment(deploy_name, namespace)
        ready   = dep.status.ready_replicas or 0
        desired = dep.spec.replicas or replicas
        logger.info(f"[{attempt+1}/120] ready={ready}/{desired}")

        if ready >= 1:
            logger.info(f"✅ Deployment {deploy_name} is Ready! ({ready}/{desired} replicas)")
            logger.info(f"   Inference : {serving_url}")
            logger.info(f"   Metrics   : http://{NODE_IP}:{metrics_node_port}/metrics")
            if hpa_enabled:
                logger.info(f"   HPA       : {hpa_name}  cpu={hpa_cpu_target}%  [{hpa_min_replicas},{hpa_max_replicas}]")
            return

    raise TimeoutError(f"Deployment {deploy_name} did not reach Ready state within 10 minutes")

def save_serving_job_to_db(researcher_id, model_name, **ctx):
    import psycopg2  # ← missing import

    # ── DB connection vars (were never defined) ──────────────────────────────
    db_host = os.environ.get("MLFLOW_DB_HOST", "mlflow-postgresql-svc.mlops-mlflow.svc.cluster.local")
    db_port = os.environ.get("MLFLOW_DB_PORT", "5432")
    db_name = os.environ.get("MLFLOW_DB_NAME", "mlflow")
    db_user = os.environ.get("MLFLOW_DB_USER", "mlflow")
    db_pass = os.environ.get("MLFLOW_DB_PASSWORD", "mlflow-password")

    ti             = ctx["ti"]
    pod_name       = ti.xcom_pull(task_ids="deploy_serving_pod", key="pod_name")
    svc_name       = ti.xcom_pull(task_ids="deploy_serving_pod", key="svc_name")
    serving_url    = ti.xcom_pull(task_ids="deploy_serving_pod", key="serving_url")
    dag_run_id     = ctx["dag_run"].run_id
    version        = ti.xcom_pull(task_ids="validate_and_promote", key="model_version") or "latest"
    replicas       = ti.xcom_pull(task_ids="deploy_serving_pod", key="replicas") or 1
    hpa_enabled    = ti.xcom_pull(task_ids="deploy_serving_pod", key="hpa_enabled") or False
    hpa_min        = ti.xcom_pull(task_ids="deploy_serving_pod", key="hpa_min_replicas")
    hpa_max        = ti.xcom_pull(task_ids="deploy_serving_pod", key="hpa_max_replicas")
    hpa_cpu        = ti.xcom_pull(task_ids="deploy_serving_pod", key="hpa_cpu_target")

    conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name,
        user=db_user, password=db_pass, connect_timeout=10,
        options="-c lc_messages=C",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO researcher_serving_jobs
                    (researcher_id, model_name, version, dag_run_id,
                     pod_name, svc_name, state, ready, serving_url,
                     replicas, hpa_enabled, hpa_min_replicas, hpa_max_replicas, hpa_cpu_target)
                VALUES (%s, %s, %s, %s, %s, %s, 'success', true, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dag_run_id) DO UPDATE
                    SET state            = 'success',
                        ready            = true,
                        pod_name         = EXCLUDED.pod_name,
                        svc_name         = EXCLUDED.svc_name,
                        serving_url      = EXCLUDED.serving_url,
                        replicas         = EXCLUDED.replicas,
                        hpa_enabled      = EXCLUDED.hpa_enabled,
                        hpa_min_replicas = EXCLUDED.hpa_min_replicas,
                        hpa_max_replicas = EXCLUDED.hpa_max_replicas,
                        hpa_cpu_target   = EXCLUDED.hpa_cpu_target,
                        updated_at       = NOW()
            """, (researcher_id, model_name, version, dag_run_id,
                  pod_name, svc_name, serving_url,
                  replicas, hpa_enabled, hpa_min, hpa_max, hpa_cpu))
        conn.commit()
        logger.info("✅ serving job saved to DB for dag_run_id=%s", dag_run_id)
    finally:
        conn.close()
        
with DAG(
    dag_id="serve_model_dag",
    default_args={"owner": "mlops", "retries": 1},
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["serving", "mlflow"],
) as dag:

    researcher_id    = "{{ dag_run.conf['researcher_id'] }}"
    model_name       = "{{ dag_run.conf['model_name'] }}"
    version          = "{{ dag_run.conf.get('version', 'latest') }}"
    threshold        = "{{ dag_run.conf.get('threshold', 0.0) }}"
    replicas         = "{{ dag_run.conf.get('replicas', 1) }}"
    hpa_enabled      = "{{ dag_run.conf.get('hpa_enabled', False) }}"
    hpa_min_replicas = "{{ dag_run.conf.get('hpa_min_replicas', 1) }}"
    hpa_max_replicas = "{{ dag_run.conf.get('hpa_max_replicas', 5) }}"
    hpa_cpu_target   = "{{ dag_run.conf.get('hpa_cpu_target', 70) }}"

    t1 = PythonOperator(
        task_id="validate_and_promote",
        python_callable=validate_and_promote,
        op_kwargs={
            "researcher_id": researcher_id,
            "model_name":    model_name,
            "version":       version,
            "threshold":     threshold,
        },
    )

    t2 = PythonOperator(
        task_id="deploy_serving_pod",
        python_callable=launch_and_wait_healthy,
        op_kwargs={
            "researcher_id":    researcher_id,
            "model_name":       model_name,
            "replicas":         replicas,
            "hpa_enabled":      hpa_enabled,
            "hpa_min_replicas": hpa_min_replicas,
            "hpa_max_replicas": hpa_max_replicas,
            "hpa_cpu_target":   hpa_cpu_target,
        },
    )

    t3 = PythonOperator(
        task_id="save_serving_job",
        python_callable=save_serving_job_to_db,
        op_kwargs={
            "researcher_id": researcher_id,
            "model_name":    model_name,
        },
    )

    t1 >> t2 >> t3