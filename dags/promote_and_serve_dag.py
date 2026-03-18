from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
from mlflow import MlflowClient
import os
import time
import logging

logger = logging.getLogger(__name__)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000")
ADMIN_USER = "admin"
ADMIN_PASS = "mlops-admin-2024"


def validate_and_promote(researcher_id, model_name, version, threshold, **ctx):
    import mlflow

    logger.info(f"=== VALIDATE | researcher={researcher_id} model={model_name} v={version} ===")

    os.environ["MLFLOW_TRACKING_URI"]      = MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = ADMIN_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = ADMIN_PASS

    mlflow.set_tracking_uri(MLFLOW_URI)
    admin_client = MlflowClient(tracking_uri=MLFLOW_URI)

    # ── Get model version ─────────────────────────────────────────────────────
    if version == "latest":
        versions = admin_client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        mv = versions[0]
    else:
        mv = admin_client.get_model_version(model_name, version)

    logger.info(f"Found model version: {mv.version} run_id={mv.run_id}")

    # ── Validate accuracy ─────────────────────────────────────────────────────
    run    = admin_client.get_run(mv.run_id)
    acc    = float(run.data.metrics.get("accuracy", 0))
    thresh = float(threshold)

    logger.info(f"accuracy={acc}, threshold={thresh}")
    if acc < thresh:
        raise ValueError(f"accuracy {acc} is below threshold {thresh}")

    # ── Promote to production ─────────────────────────────────────────────────
    admin_client.set_registered_model_alias(model_name, "production", mv.version)
    logger.info(f"✓ Alias 'production' → {model_name} v{mv.version}")

    ctx["ti"].xcom_push(key="model_version", value=mv.version)
    logger.info("=== VALIDATE DONE ===")


def launch_and_wait_healthy(model_name, **ctx):
    from kubernetes import client as k8s, config as k8s_config

    logger.info(f"=== DEPLOY SERVING POD | model={model_name} ===")

    k8s_config.load_incluster_config()
    core_v1   = k8s.CoreV1Api()
    namespace = "mlops-serving"
    pod_name  = f"serve-{model_name.replace('_', '-').lower()}"
    svc_name  = f"{pod_name}-svc"

    # ── Delete old pod if exists ──────────────────────────────────────────────
    try:
        core_v1.delete_namespaced_pod(pod_name, namespace)
        logger.info(f"Deleted old pod {pod_name}, waiting 3s...")
        time.sleep(3)
    except k8s.exceptions.ApiException as e:
        if e.status != 404:
            raise

    # ── Create pod ────────────────────────────────────────────────────────────
    pod = k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(
            name=pod_name,
            namespace=namespace,
            labels={"app": pod_name, "model": model_name},
        ),
        spec=k8s.V1PodSpec(
            restart_policy="Always",
            containers=[
                k8s.V1Container(
                    name="mlflow-serve",
                    image="ghcr.io/mlflow/mlflow:v3.10.1",
                    command=["sh", "-c"],
                    args=[
                        f"pip install mlflow boto3 scikit-learn --quiet && "
                        f"mlflow models serve "
                        f"--model-uri 'models:/{model_name}@production' "
                        f"--host 0.0.0.0 --port 8001 --no-conda"
                    ],
                    ports=[k8s.V1ContainerPort(container_port=8001)],
                    env=[
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_URI",      value=MLFLOW_URI),
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_USERNAME", value=ADMIN_USER),
                        k8s.V1EnvVar(name="MLFLOW_TRACKING_PASSWORD", value=ADMIN_PASS),
                        k8s.V1EnvVar(name="AWS_ENDPOINT_URL",         value="http://minio-svc.mlops-minio.svc.cluster.local:9000"),
                        k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID",        value="minio-admin"),
                        k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY",    value="minio-admin"),
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

    core_v1.create_namespaced_pod(namespace, pod)
    logger.info(f"✓ Pod {pod_name} created")

    # ── Create / update Service ───────────────────────────────────────────────
    service = k8s.V1Service(
        metadata=k8s.V1ObjectMeta(name=svc_name, namespace=namespace),
        spec=k8s.V1ServiceSpec(
            selector={"app": pod_name},
            ports=[k8s.V1ServicePort(port=8001, target_port=8001)],
        ),
    )
    try:
        core_v1.read_namespaced_service(svc_name, namespace)
        core_v1.replace_namespaced_service(svc_name, namespace, service)
        logger.info(f"✓ Service {svc_name} updated")
    except k8s.exceptions.ApiException as e:
        if e.status == 404:
            core_v1.create_namespaced_service(namespace, service)
            logger.info(f"✓ Service {svc_name} created")
        else:
            raise

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
                logger.info(f"   Internal : http://{svc_name}.{namespace}.svc.cluster.local:8001/invocations")
                logger.info(f"   Port-fwd : kubectl port-forward svc/{svc_name} 8001:8001 -n {namespace}")
                return   # ← DAG task exits, pod keeps running

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
            "model_name": model_name,
        },
    )

    t1 >> t2