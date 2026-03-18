from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import mlflow
from mlflow import MlflowClient
import subprocess
import os
import logging

logger  = logging.getLogger(__name__)
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000")

# MLflow admin — needed to READ metrics since researcher has NO_PERMISSIONS on other data
ADMIN_USER = "admin"
ADMIN_PASS = "mlops-admin-2024"


def validate_and_promote(researcher_id, model_name, version, threshold, **ctx):
    logger.info(f"=== VALIDATE | researcher={researcher_id} model={model_name} v={version} ===")

    # ✅ Set admin creds via env vars — works in all MLflow versions
    os.environ["MLFLOW_TRACKING_URI"]      = MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = ADMIN_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = ADMIN_PASS

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    admin_client = MlflowClient(tracking_uri=MLFLOW_URI)

    # Get the model version
    if version == "latest":
        versions = admin_client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        mv = versions[0]
    else:
        mv = admin_client.get_model_version(model_name, version)

    run    = admin_client.get_run(mv.run_id)
    acc    = float(run.data.metrics.get("accuracy", 0))
    thresh = float(threshold)

    logger.info(f"accuracy={acc}, threshold={thresh}")
    if acc < thresh:
        raise ValueError(f"accuracy {acc} is below threshold {thresh}")

    admin_client.set_registered_model_alias(model_name, "staging", mv.version)
    logger.info(f"✓ Alias 'staging' → {model_name} v{mv.version}")

    ctx["ti"].xcom_push(key="model_version", value=mv.version)
    ctx["ti"].xcom_push(key="model_uri",     value=f"models:/{model_name}@staging")


def deploy_serving_pod(researcher_id, model_name, **ctx):
    logger.info(f"=== DEPLOY SERVING POD | researcher={researcher_id} model={model_name} ===")

    version   = ctx["ti"].xcom_pull(key="model_version")
    model_uri = ctx["ti"].xcom_pull(key="model_uri")

    # Safe k8s name: researcher_abc/iris-classifier → researcher-abc-iris-classifier
    safe_name = f"{researcher_id}-{model_name}".replace("_", "-").replace("/", "-").lower()
    namespace = "mlops-serving"

    # Dynamic deployment manifest
    manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: serve-{safe_name}
  namespace: {namespace}
  labels:
    app: serve-{safe_name}
    researcher: {researcher_id}
    model: {model_name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: serve-{safe_name}
  template:
    metadata:
      labels:
        app: serve-{safe_name}
        researcher: {researcher_id}
    spec:
      containers:
        - name: mlflow-serve
          image: ghcr.io/mlflow/mlflow:v3.10.1
          command:
            - sh
            - -c
            - |
              pip install mlflow boto3 scikit-learn --quiet && \\
              mlflow models serve \\
                --model-uri "{model_uri}" \\
                --host 0.0.0.0 \\
                --port 8001 \\
                --no-conda
          ports:
            - containerPort: 8001
          env:
            - name: MLFLOW_TRACKING_URI
              value: {MLFLOW_URI}
            - name: MLFLOW_TRACKING_USERNAME
              value: {researcher_id}
            - name: AWS_ENDPOINT_URL
              value: http://minio-svc.mlops-minio.svc.cluster.local:9000
            - name: AWS_ACCESS_KEY_ID
              value: minio-admin
            - name: AWS_SECRET_ACCESS_KEY
              value: minio-admin
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: serve-{safe_name}-svc
  namespace: {namespace}
spec:
  selector:
    app: serve-{safe_name}
  ports:
    - port: 8001
      targetPort: 8001
"""

    # Write manifest and apply
    manifest_path = f"/tmp/serve-{safe_name}.yaml"
    with open(manifest_path, "w") as f:
        f.write(manifest)

    # Ensure namespace exists
    subprocess.run(
        ["kubectl", "create", "namespace", namespace, "--dry-run=client", "-o", "yaml"],
        capture_output=True
    )
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=f"apiVersion: v1\nkind: Namespace\nmetadata:\n  name: {namespace}",
        text=True, capture_output=True
    )

    result = subprocess.run(
        ["kubectl", "apply", "-f", manifest_path],
        capture_output=True, text=True
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl apply failed:\n{result.stderr}")

    # Promote alias to production
    os.environ["MLFLOW_TRACKING_USERNAME"] = ADMIN_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = ADMIN_PASS
    admin_client = MlflowClient(tracking_uri=MLFLOW_URI)
    admin_client.set_registered_model_alias(model_name, "production", version)
    logger.info(f"✓ Alias 'production' → {model_name} v{version}")

    logger.info(f"✅ Serving pod deployed: serve-{safe_name} in {namespace}")
    logger.info(f"   Endpoint: http://serve-{safe_name}-svc.{namespace}.svc.cluster.local:8001/invocations")


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
        python_callable=deploy_serving_pod,
        op_kwargs={
            "researcher_id": researcher_id,
            "model_name":    model_name,
        },
    )

    t1 >> t2