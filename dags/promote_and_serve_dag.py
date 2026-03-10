from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import subprocess
import os

default_args = {"owner": "mlops", "retries": 1}

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000")

def validate_metrics(run_id, threshold, **ctx):
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_URI)
    run = client.get_run(run_id)
    accuracy = float(run.data.metrics.get("accuracy", 0))
    print(f"accuracy={accuracy}, threshold={threshold}")
    if accuracy < float(threshold):
        raise ValueError(f"accuracy {accuracy} below threshold {threshold}")
    ctx["ti"].xcom_push(key="accuracy", value=accuracy)

def register_model(run_id, **ctx):
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_URI)
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name="MyModel",
    )
    print(f"Registered MyModel v{result.version}")
    ctx["ti"].xcom_push(key="model_version", value=result.version)

def transition_to_staging(**ctx):
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_URI)
    version = ctx["ti"].xcom_pull(key="model_version")
    client.transition_model_version_stage(
        name="MyModel", version=version, stage="Staging"
    )

def transition_to_production(**ctx):
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_URI)
    version = ctx["ti"].xcom_pull(key="model_version")
    client.transition_model_version_stage(
        name="MyModel", version=version, stage="Production",
        archive_existing_versions=True,
    )

def deploy_serve_pod(**ctx):
    subprocess.run(
        ["kubectl", "apply", "-f", "/opt/airflow/serve/deployment.yaml"],
        check=True
    )

with DAG(
    dag_id="promote_and_serve_dag",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlflow", "serving"],
) as dag:

    run_id   = "{{ dag_run.conf['run_id'] }}"
    threshold = "{{ dag_run.conf.get('threshold', 0.90) }}"

    t1 = PythonOperator(
        task_id="validate_metrics",
        python_callable=validate_metrics,
        op_kwargs={"run_id": run_id, "threshold": threshold},
    )

    t2 = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        op_kwargs={"run_id": run_id},
    )

    t3 = PythonOperator(
        task_id="transition_to_staging",
        python_callable=transition_to_staging,
    )

    t4 = PythonOperator(
        task_id="transition_to_production",
        python_callable=transition_to_production,
    )

    t5 = PythonOperator(
        task_id="deploy_serve_pod",
        python_callable=deploy_serve_pod,
    )

    t1 >> t2 >> t3 >> t4 >> t5