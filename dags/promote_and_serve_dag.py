from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
from mlflow import MlflowClient
import os
import time
import logging

logger = logging.getLogger(__name__)

MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000")
ADMIN_USER   = "admin"
ADMIN_PASS   = "mlops-admin-2024"

# ── Node access ───────────────────────────────────────────────────────────────
# NODE_IP: the IP of any Kubernetes node.
# The DAG creates a NodePort service so the pod is reachable at
# http://NODE_IP:<nodePort>/invocations from outside the cluster.
# NodePort is auto-allocated in range 30000-32767 based on model name.
NODE_IP = os.environ.get("NODE_IP", "192.168.0.64")

# ── Traefik config (optional — used only if Traefik CRDs are present) ─────────
TRAEFIK_HOST       = os.environ.get("TRAEFIK_HOST",        NODE_IP)
TRAEFIK_ENTRYPOINT = os.environ.get("TRAEFIK_ENTRYPOINT",  "web")
TRAEFIK_GROUP      = os.environ.get("TRAEFIK_CRD_GROUP",   "traefik.io")


def _serving_path(researcher_id: str, model_name: str) -> str:
    """URL path prefix used by Traefik to route to this model's service."""
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


def _create_or_replace_middleware(
    custom_api,
    namespace: str,
    name: str,
    strip_prefix: str,
) -> None:
    """
    Create (or replace) a Traefik StripPrefix middleware.
    Works with both traefik.io (v3) and traefik.containo.us (v2).
    """
    body = {
        "apiVersion": f"{TRAEFIK_GROUP}/v1alpha1",
        "kind": "Middleware",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "stripPrefix": {
                "prefixes": [strip_prefix],
                "forceSlash": False,
            }
        },
    }
    try:
        custom_api.delete_namespaced_custom_object(
            group=TRAEFIK_GROUP, version="v1alpha1",
            namespace=namespace, plural="middlewares", name=name,
        )
        logger.info(f"Deleted old middleware: {name}")
        time.sleep(1)
    except Exception:
        pass  # doesn't exist yet

    custom_api.create_namespaced_custom_object(
        group=TRAEFIK_GROUP, version="v1alpha1",
        namespace=namespace, plural="middlewares", body=body,
    )
    logger.info(f"✓ Middleware created: {name}  strip={strip_prefix}")


def _create_or_replace_ingressroute(
    custom_api,
    namespace: str,
    name: str,
    svc_name: str,
    path_prefix: str,
    middleware_name: str,
) -> None:
    """
    Create (or replace) a Traefik IngressRoute for the serving pod.

    Route: PathPrefix(`<path_prefix>`) → svc_name:8001
    After stripping the prefix the container receives the request at /invocations.
    """
    # Build the host+path match rule.
    # If TRAEFIK_HOST is set to an IP or hostname, scope to that host.
    if TRAEFIK_HOST and TRAEFIK_HOST not in ("", "0.0.0.0"):
        match_rule = f"Host(`{TRAEFIK_HOST}`) && PathPrefix(`{path_prefix}`)"
    else:
        match_rule = f"PathPrefix(`{path_prefix}`)"

    body = {
        "apiVersion": f"{TRAEFIK_GROUP}/v1alpha1",
        "kind": "IngressRoute",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "entryPoints": [TRAEFIK_ENTRYPOINT],
            "routes": [
                {
                    "match": match_rule,
                    "kind": "Rule",
                    "services": [
                        {"name": svc_name, "port": 8001}
                    ],
                    "middlewares": [
                        {"name": middleware_name, "namespace": namespace}
                    ],
                }
            ],
        },
    }

    try:
        custom_api.delete_namespaced_custom_object(
            group=TRAEFIK_GROUP, version="v1alpha1",
            namespace=namespace, plural="ingressroutes", name=name,
        )
        logger.info(f"Deleted old IngressRoute: {name}")
        time.sleep(1)
    except Exception:
        pass

    custom_api.create_namespaced_custom_object(
        group=TRAEFIK_GROUP, version="v1alpha1",
        namespace=namespace, plural="ingressroutes", body=body,
    )
    logger.info(f"✓ IngressRoute created: {name}  match={match_rule}")


def launch_and_wait_healthy(researcher_id, model_name, **ctx):
    from kubernetes import client as k8s

    logger.info(f"=== DEPLOY SERVING POD | model={model_name} researcher={researcher_id} ===")

    core_v1, custom_api = _k8s_clients()

    namespace    = "mlops-serving"
    model_slug   = model_name.replace("_", "-").lower()
    rid_slug     = researcher_id.replace("_", "-").lower()
    pod_name     = f"serve-{model_slug}"
    svc_name     = f"{pod_name}-svc"
    mw_name      = f"strip-{pod_name}"
    ir_name      = f"ingress-{pod_name}"
    path_prefix  = _serving_path(researcher_id, model_name)

    # serving_url is set after NodePort service is created below

    # ── Pod spec ──────────────────────────────────────────────────────────────
    # Each model has its own permanent pod. We never delete it.
    # create_namespaced_pod is skipped if the pod already exists (409 → ignore).
    pod = k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(
            name=pod_name,
            namespace=namespace,
            labels={"app": pod_name, "model": model_slug, "researcher": rid_slug},
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

    try:
        core_v1.create_namespaced_pod(namespace, pod)
        logger.info(f"✓ Pod {pod_name} created")
    except k8s.exceptions.ApiException as e:
        if e.status == 409:
            logger.info(f"✓ Pod {pod_name} already exists — reusing it")
        else:
            raise

    # ── Create / update NodePort Service ─────────────────────────────────────
    # NodePort exposes the pod on every cluster node at node_port.
    # Access: http://<NODE_IP>:<node_port>/invocations  — no port-forward needed.
    # NodePort range is 30000-32767. We derive a stable port from the model name.
    # hashlib.md5 gives a stable value across Python restarts.
    # Python's built-in hash() is randomized per-process (PYTHONHASHSEED)
    # so it would assign a different NodePort every time Airflow restarts.
    import hashlib
    node_port = 30000 + (int(hashlib.md5(model_name.encode()).hexdigest(), 16) % 2767)

    service = k8s.V1Service(
        metadata=k8s.V1ObjectMeta(
            name=svc_name,
            namespace=namespace,
            labels={"app": pod_name, "model": model_slug, "researcher": rid_slug},
        ),
        spec=k8s.V1ServiceSpec(
            type="NodePort",
            selector={"app": pod_name},
            ports=[k8s.V1ServicePort(
                port=8001,
                target_port=8001,
                node_port=node_port,
            )],
        ),
    )
    try:
        core_v1.read_namespaced_service(svc_name, namespace)
        # NodePort cannot be patched in-place when type changes — delete and recreate
        core_v1.delete_namespaced_service(svc_name, namespace)
        logger.info(f"Deleted old service {svc_name}")
        time.sleep(1)
    except k8s.exceptions.ApiException as e:
        if e.status != 404:
            raise

    core_v1.create_namespaced_service(namespace, service)
    logger.info(f"✓ NodePort service {svc_name} created  nodePort={node_port}")

    # Serving URL is now directly accessible via NodePort — no port-forward needed
    serving_url = f"http://{NODE_IP}:{node_port}/invocations"
    logger.info(f"✓ Serving URL: {serving_url}")

    # ── Create Traefik StripPrefix Middleware ─────────────────────────────────
    try:
        _create_or_replace_middleware(
            custom_api=custom_api,
            namespace=namespace,
            name=mw_name,
            strip_prefix=path_prefix,
        )
    except Exception as exc:
        # Non-fatal: Traefik CRDs might not be installed — log and continue.
        # The pod will still be reachable via kubectl port-forward.
        logger.warning(f"Could not create Middleware (Traefik CRDs installed?): {exc}")

    # ── Create Traefik IngressRoute ───────────────────────────────────────────
    try:
        _create_or_replace_ingressroute(
            custom_api=custom_api,
            namespace=namespace,
            name=ir_name,
            svc_name=svc_name,
            path_prefix=path_prefix,
            middleware_name=mw_name,
        )
        logger.info(f"✓ Serving endpoint: http://{TRAEFIK_HOST}{path_prefix}/invocations")
    except Exception as exc:
        logger.warning(f"Could not create IngressRoute (Traefik CRDs installed?): {exc}")

    # ── Push final URL to XCom ────────────────────────────────────────────────
    ctx["ti"].xcom_push(key="serving_url",  value=serving_url)
    ctx["ti"].xcom_push(key="path_prefix",  value=path_prefix)
    ctx["ti"].xcom_push(key="pod_name",     value=pod_name)
    ctx["ti"].xcom_push(key="svc_name",     value=svc_name)
    ctx["ti"].xcom_push(key="host_port",    value=str(host_port))
    ctx["ti"].xcom_push(key="node_ip",      value=NODE_IP)

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
                logger.info(f"   Direct URL  : {serving_url}")
                logger.info(f"   hostPort    : {host_port} on {NODE_IP}")
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