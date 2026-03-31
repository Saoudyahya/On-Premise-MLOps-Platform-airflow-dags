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

TRAEFIK_HOST       = os.environ.get("TRAEFIK_HOST",        NODE_IP)
TRAEFIK_ENTRYPOINT = os.environ.get("TRAEFIK_ENTRYPOINT",  "web")
TRAEFIK_GROUP      = os.environ.get("TRAEFIK_CRD_GROUP",   "traefik.io")


def _serving_path(researcher_id: str, model_name: str) -> str:
    model_slug = model_name.replace("_", "-").lower()
    rid_slug   = researcher_id.replace("_", "-").lower()
    return f"/serving/{rid_slug}/{model_slug}"


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

    core_v1, custom_api = _k8s_clients()

    namespace   = "mlops-serving"
    model_slug  = model_name.replace("_", "-").lower()
    rid_slug    = researcher_id.replace("_", "-").lower()
    pod_name    = f"serve-{model_slug}"
    svc_name    = f"{pod_name}-svc"
    path_prefix = _serving_path(researcher_id, model_name)

    # ── Stable NodePort for inference (8001) ─────────────────────────────────
    node_port = 30000 + (int(hashlib.md5(model_name.encode()).hexdigest(), 16) % 2767)

    # ── Stable NodePort for Prometheus metrics (8002) ─────────────────────────
    # Offset by 1000 so it never collides with the inference port range
    metrics_node_port = 30000 + (int(hashlib.md5((model_name + "_metrics").encode()).hexdigest(), 16) % 2767)
    # Ensure they don't collide (astronomically unlikely but guard anyway)
    if metrics_node_port == node_port:
        metrics_node_port = node_port + 1 if node_port < 32767 else node_port - 1

    serving_url = f"http://{NODE_IP}:{node_port}/invocations"
    logger.info(f"Inference NodePort : {node_port}  → {serving_url}")
    logger.info(f"Metrics  NodePort  : {metrics_node_port} → http://{NODE_IP}:{metrics_node_port}/metrics")

    # ── Pod spec ──────────────────────────────────────────────────────────────
    # Key change: prometheus_flask_exporter is added to the pip install so that
    # MLflow's Flask app automatically exposes /metrics on port 8002.
    # The exporter is auto-detected via the PROMETHEUS_MULTIPROC_DIR env var
    # and the FLASK_APP hook — no code changes to MLflow are needed.
    pod = k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(
            name=pod_name,
            namespace=namespace,
            labels={
                "app":        pod_name,
                "model":      model_slug,
                "researcher": rid_slug,
                # Prometheus ServiceMonitor selector label
                "prometheus-scrape": "true",
            },
            annotations={
                # Prometheus pod-level auto-discovery annotations (works without ServiceMonitor)
                "prometheus.io/scrape": "true",
                "prometheus.io/port":   "8002",
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
                        # Install prometheus_flask_exporter alongside the existing deps.
                        # It auto-instruments Flask when imported, exposing /metrics on the
                        # same process.  We bind the metrics endpoint to port 8002 via the
                        # PROMETHEUS_FLASK_EXPORTER_PORT env var so it doesn't clash with
                        # the inference port 8001.
                        "pip install mlflow boto3 scikit-learn prometheus-flask-exporter --quiet && "
                        "mlflow models serve "
                        f"--model-uri 'models:/{model_name}@production' "
                        "--host 0.0.0.0 --port 8001 --no-conda"
                    ],
                    ports=[
                        k8s.V1ContainerPort(container_port=8001, name="inference"),
                        k8s.V1ContainerPort(container_port=8002, name="metrics"),
                    ],
                    env=[
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_URI",      value=MLFLOW_URI),
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_USERNAME", value=ADMIN_USER),
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_PASSWORD", value=ADMIN_PASS),
                        k8s.V1EnvVar(name="AWS_ENDPOINT_URL",         value="http://minio-svc.mlops-minio.svc.cluster.local:9000"),
                        k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID",        value="minio-admin"),
                        k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY",    value="minio-admin"),
                        # Tell prometheus_flask_exporter which port to bind /metrics on.
                        # When this env var is set the exporter starts a separate WSGI
                        # server on 8002 so /metrics never conflicts with /invocations.
                        k8s.V1EnvVar(name="PROMETHEUS_FLASK_EXPORTER_PORT", value="8002"),
                    ],
                    readiness_probe=k8s.V1Probe(
                        http_get=k8s.V1HTTPGetAction(path="/health", port=8001),
                        initial_delay_seconds=15,
                        period_seconds=5,
                        failure_threshold=10,
                    ),
                    resources=k8s.V1ResourceRequirements(
                        requests={"memory": "512Mi", "cpu": "250m"},
                        limits={"memory": "1Gi",   "cpu": "500m"},
                    ),
                )
            ],
        ),
    )

    try:
        core_v1.create_namespaced_pod(namespace, pod)
        logger.info(f"✓ Pod {pod_name} created")
    except k8s.exceptions.ApiException as e:
        if e.status == 409:
            logger.info(f"✓ Pod {pod_name} already exists — reusing it")
        else:
            raise

    # ── Service: expose both inference (8001) and metrics (8002) ports ───────
    service = k8s.V1Service(
        metadata=k8s.V1ObjectMeta(
            name=svc_name,
            namespace=namespace,
            labels={
                "app":        pod_name,
                "model":      model_slug,
                "researcher": rid_slug,
                "prometheus-scrape": "true",
            },
        ),
        spec=k8s.V1ServiceSpec(
            type="NodePort",
            selector={"app": pod_name},
            ports=[
                k8s.V1ServicePort(
                    name="inference",
                    port=8001,
                    target_port=8001,
                    node_port=node_port,
                ),
                k8s.V1ServicePort(
                    name="metrics",
                    port=8002,
                    target_port=8002,
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
    logger.info(f"✓ NodePort service {svc_name} created  inferencePort={node_port}  metricsPort={metrics_node_port}")

    # ── Push to XCom ──────────────────────────────────────────────────────────
    ctx["ti"].xcom_push(key="serving_url",       value=serving_url)
    ctx["ti"].xcom_push(key="metrics_url",       value=f"http://{NODE_IP}:{metrics_node_port}/metrics")
    ctx["ti"].xcom_push(key="metrics_node_port", value=metrics_node_port)
    ctx["ti"].xcom_push(key="node_port",         value=node_port)
    ctx["ti"].xcom_push(key="pod_name",          value=pod_name)
    ctx["ti"].xcom_push(key="svc_name",          value=svc_name)
    ctx["ti"].xcom_push(key="node_ip",           value=NODE_IP)

    # ── Poll until Running + Ready ────────────────────────────────────────────
    logger.info("Waiting for pod to become Ready (max 5 min)...")
    for attempt in range(60):
        time.sleep(5)
        pod_status = core_v1.read_namespaced_pod(pod_name, namespace)
        phase      = pod_status.status.phase

        if phase == "Running":
            conditions = pod_status.status.conditions or []
            ready      = any(c.type == "Ready" and c.status == "True" for c in conditions)
            if ready:
                logger.info(f"✅ Pod {pod_name} is Running and Ready!")
                logger.info(f"   Inference URL : {serving_url}")
                logger.info(f"   Metrics URL   : http://{NODE_IP}:{metrics_node_port}/metrics")
                return
        elif phase in ("Failed", "Unknown"):
            raise RuntimeError(f"Pod {pod_name} entered phase: {phase}")

        logger.info(f"[{attempt+1}/60] phase={phase} — waiting...")

    raise TimeoutError(f"Pod {pod_name} did not become Ready within 5 minutes")


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

    t1 >> t2