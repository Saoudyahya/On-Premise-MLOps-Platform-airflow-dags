from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
from mlflow import MlflowClient
import os
import time
import logging
import hashlib

logger = logging.getLogger(__name__)

MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000")
ADMIN_USER   = "admin"
ADMIN_PASS   = "mlops-admin-2024"

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
    return k8s.CoreV1Api(), k8s.CustomObjectsApi()


def launch_and_wait_healthy(researcher_id, model_name, **ctx):
    from kubernetes import client as k8s

    logger.info(f"=== DEPLOY SERVING POD | model={model_name} researcher={researcher_id} ===")

    core_v1, _ = _k8s_clients()

    namespace  = "mlops-serving"
    model_slug = model_name.replace("_", "-").lower()
    rid_slug   = researcher_id.replace("_", "-").lower()
    pod_name   = f"serve-{rid_slug}-{model_slug}"
    svc_name   = f"{pod_name}-svc"

    # ── NodePort for inference (8080 — MLServer REST port) ────────────────────
    node_port = 30000 + (int(hashlib.md5(model_name.encode()).hexdigest(), 16) % 2767)

    # ── NodePort for Prometheus metrics (8082 — MLServer built-in) ───────────
    metrics_node_port = 30000 + (int(hashlib.md5((model_name + "_metrics").encode()).hexdigest(), 16) % 2767)
    if metrics_node_port == node_port:
        metrics_node_port = node_port + 1 if node_port < 32767 else node_port - 1

    # MLServer /invocations is on port 8080
    serving_url = f"http://{NODE_IP}:{node_port}/invocations"
    logger.info(f"Inference NodePort : {node_port}  → {serving_url}")
    logger.info(f"Metrics  NodePort  : {metrics_node_port} → http://{NODE_IP}:{metrics_node_port}/metrics")

    # ── Delete existing pod so we always get a fresh one with latest config ───
    try:
        core_v1.delete_namespaced_pod(pod_name, namespace)
        logger.info(f"Deleted existing pod {pod_name}, waiting for termination...")
        for _ in range(30):
            time.sleep(2)
            try:
                core_v1.read_namespaced_pod(pod_name, namespace)
            except k8s.exceptions.ApiException as e:
                if e.status == 404:
                    logger.info("Pod terminated.")
                    break
    except k8s.exceptions.ApiException as e:
        if e.status != 404:
            raise

    # ── Pod spec ──────────────────────────────────────────────────────────────
    # --enable-mlserver tells MLflow to use MLServer as the backend instead of
    # the plain Flask/uvicorn server. MLServer exposes Prometheus metrics
    # natively on port 8082 at /metrics — no extra instrumentation needed.
    #
    # MLServer metric names: mlserver_requests_total, mlserver_request_duration_seconds, etc.
    pod = k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(
            name=pod_name,
            namespace=namespace,
            labels={
                "app":               pod_name,
                "model":             f"{rid_slug}-{model_slug}",
                "researcher":        rid_slug,
                "prometheus-scrape": "true",
            },
            annotations={
                "prometheus.io/scrape": "true",
                "prometheus.io/port":   "8082",
                "prometheus.io/path":   "/metrics",
            },
        ),
        spec=k8s.V1PodSpec(
            restart_policy="Always",
            containers=[
                k8s.V1Container(
                    name="mlflow-serve",
                    image="ghcr.io/mlflow/mlflow:v3.10.1",
                    command=["sh", "-c"],
                    args=[
                        # mlserver and mlserver-mlflow provide the MLServer runtime.
                        # MLServer automatically starts /metrics on port 8082.
                        "pip install mlflow boto3 scikit-learn mlserver mlserver-mlflow --quiet && "
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
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_URI",      value=MLFLOW_URI),
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_USERNAME", value=ADMIN_USER),
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_PASSWORD", value=ADMIN_PASS),
                        k8s.V1EnvVar(name="AWS_ENDPOINT_URL",         value="http://minio-svc.mlops-minio.svc.cluster.local:9000"),
                        k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID",        value="minio-admin"),
                        k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY",    value="minio-admin"),
                        # MLServer port config
                        k8s.V1EnvVar(name="MLSERVER_HTTP_PORT",          value="8080"),
                        k8s.V1EnvVar(name="MLSERVER_GRPC_PORT",          value="8081"),
                        k8s.V1EnvVar(name="MLSERVER_METRICS_PORT",       value="8082"),
                        k8s.V1EnvVar(name="MLSERVER_MODEL_NAME",         value=model_name),
                        # Fix: MLServer parallel workers crash on Python 3.10 due to
                        # uvloop RuntimeError("There is no current event loop").
                        # Setting workers=0 disables the multiprocess pool and runs
                        # inference directly in the main async loop — fine for single-model pods.
                        k8s.V1EnvVar(name="MLSERVER_PARALLEL_WORKERS",   value="0"),
                    ],
                    readiness_probe=k8s.V1Probe(
                        http_get=k8s.V1HTTPGetAction(path="/v2/health/ready", port=8080),
                        initial_delay_seconds=30,
                        period_seconds=10,
                        failure_threshold=12,
                    ),
                    liveness_probe=k8s.V1Probe(
                        http_get=k8s.V1HTTPGetAction(path="/v2/health/live", port=8080),
                        initial_delay_seconds=60,
                        period_seconds=15,
                        failure_threshold=6,
                    ),
                    resources=k8s.V1ResourceRequirements(
                        requests={"memory": "512Mi", "cpu": "250m"},
                        limits={"memory": "2Gi",   "cpu": "1000m"},
                    ),
                )
            ],
        ),
    )

    core_v1.create_namespaced_pod(namespace, pod)
    logger.info(f"✓ Pod {pod_name} created")

    # ── Service ───────────────────────────────────────────────────────────────
    service = k8s.V1Service(
        metadata=k8s.V1ObjectMeta(
            name=svc_name,
            namespace=namespace,
            labels={
                "app":               pod_name,
                "model":             f"{rid_slug}-{model_slug}",
                "researcher":        rid_slug,
                "prometheus-scrape": "true",
            },
        ),
        spec=k8s.V1ServiceSpec(
            type="NodePort",
            selector={"app": pod_name},
            ports=[
                k8s.V1ServicePort(
                    name="inference",
                    port=8080,
                    target_port=8080,
                    node_port=node_port,
                ),
                k8s.V1ServicePort(
                    name="metrics",
                    port=8082,
                    target_port=8082,
                    node_port=metrics_node_port,
                ),
            ],
        ),
    )

    try:
        core_v1.read_namespaced_service(svc_name, namespace)
        core_v1.delete_namespaced_service(svc_name, namespace)
        logger.info(f"Deleted old service {svc_name}")
        time.sleep(1)
    except k8s.exceptions.ApiException as e:
        if e.status != 404:
            raise

    core_v1.create_namespaced_service(namespace, service)
    logger.info(f"✓ Service {svc_name} created  inference={node_port}  metrics={metrics_node_port}")

    # ── XCom ─────────────────────────────────────────────────────────────────
    ctx["ti"].xcom_push(key="serving_url",       value=serving_url)
    ctx["ti"].xcom_push(key="metrics_url",       value=f"http://{NODE_IP}:{metrics_node_port}/metrics")
    ctx["ti"].xcom_push(key="metrics_node_port", value=metrics_node_port)
    ctx["ti"].xcom_push(key="node_port",         value=node_port)
    ctx["ti"].xcom_push(key="pod_name",          value=pod_name)
    ctx["ti"].xcom_push(key="svc_name",          value=svc_name)
    ctx["ti"].xcom_push(key="node_ip",           value=NODE_IP)

    # ── Poll until Ready ──────────────────────────────────────────────────────
    logger.info("Waiting for pod to become Ready (max 10 min)...")
    for attempt in range(120):
        time.sleep(5)
        pod_status = core_v1.read_namespaced_pod(pod_name, namespace)
        phase      = pod_status.status.phase

        if phase == "Running":
            conditions = pod_status.status.conditions or []
            ready      = any(c.type == "Ready" and c.status == "True" for c in conditions)
            if ready:
                logger.info(f"✅ Pod {pod_name} is Running and Ready!")
                logger.info(f"   Inference : {serving_url}")
                logger.info(f"   Metrics   : http://{NODE_IP}:{metrics_node_port}/metrics")
                return
        elif phase in ("Failed", "Unknown"):
            raise RuntimeError(f"Pod {pod_name} entered phase: {phase}")

        logger.info(f"[{attempt+1}/120] phase={phase} — waiting...")

    raise TimeoutError(f"Pod {pod_name} did not become Ready within 10 minutes")


def save_serving_job_to_db(researcher_id, model_name, **ctx):
    """Persist the serving job to the MLflow PostgreSQL DB so the UI can list it."""
    import psycopg2

    ti          = ctx["ti"]
    pod_name    = ti.xcom_pull(task_ids="deploy_serving_pod", key="pod_name")
    svc_name    = ti.xcom_pull(task_ids="deploy_serving_pod", key="svc_name")
    serving_url = ti.xcom_pull(task_ids="deploy_serving_pod", key="serving_url")
    dag_run_id  = ctx["dag_run"].run_id
    version     = ti.xcom_pull(task_ids="validate_and_promote", key="model_version") or "latest"

    db_host = os.environ.get("MLFLOW_DB_HOST", "mlflow-postgresql-svc.mlops-mlflow.svc.cluster.local")
    db_port = os.environ.get("MLFLOW_DB_PORT", "5432")
    db_name = os.environ.get("MLFLOW_DB_NAME", "mlflow")
    db_user = os.environ.get("MLFLOW_DB_USER", "mlflow")
    db_pass = os.environ.get("MLFLOW_DB_PASSWORD", "mlflow-password")

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
                     pod_name, svc_name, state, ready, serving_url)
                VALUES (%s, %s, %s, %s, %s, %s, 'success', true, %s)
                ON CONFLICT (dag_run_id) DO UPDATE
                    SET state       = 'success',
                        ready       = true,
                        pod_name    = EXCLUDED.pod_name,
                        svc_name    = EXCLUDED.svc_name,
                        serving_url = EXCLUDED.serving_url,
                        updated_at  = NOW()
            """, (researcher_id, model_name, version, dag_run_id,
                  pod_name, svc_name, serving_url))
        conn.commit()
        logger.info(f"✓ Serving job saved: dag_run_id={dag_run_id} pod={pod_name}")
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

    researcher_id = "{{ dag_run.conf['researcher_id'] }}"
    model_name    = "{{ dag_run.conf['model_name'] }}"
    version       = "{{ dag_run.conf.get('version', 'latest') }}"
    threshold     = "{{ dag_run.conf.get('threshold', 0.0) }}"

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
            "researcher_id": researcher_id,
            "model_name":    model_name,
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