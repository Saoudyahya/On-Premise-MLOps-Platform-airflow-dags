from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime
from mlflow import MlflowClient
import os
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

    t2 = KubernetesPodOperator(
        task_id="deploy_serving_pod",
        name="mlflow-serving-pod",
        namespace="mlops-serving",
        image="ghcr.io/mlflow/mlflow:v3.10.1",
        cmds=["sh", "-c"],
        arguments=[
            "pip install mlflow boto3 scikit-learn --quiet && "
            "mlflow models serve "
            "--model-uri 'models://{{ dag_run.conf[\"model_name\"] }}@production' "
            "--host 0.0.0.0 --port 8001 --no-conda"
        ],
        env_vars={
            "MLFLOW_TRACKING_URI":      MLFLOW_URI,
            "MLFLOW_TRACKING_USERNAME": ADMIN_USER,
            "MLFLOW_TRACKING_PASSWORD": ADMIN_PASS,
            "AWS_ENDPOINT_URL":         "http://minio-svc.mlops-minio.svc.cluster.local:9000",
            "AWS_ACCESS_KEY_ID":        "minio-admin",
            "AWS_SECRET_ACCESS_KEY":    "minio-admin",
        },
        is_delete_operator_pod=False,   # ← keep pod alive after task finishes
        get_logs=True,
        do_xcom_push=False,
    )

    t1 >> t2