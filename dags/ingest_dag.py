from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os
import logging

logger = logging.getLogger(__name__)
default_args = {"owner": "mlops", "retries": 0}


def dvc_add(dataset, researcher_id, **ctx):
    logger.info(f"Starting dvc_add for {researcher_id}/{dataset}")
    logger.info(f"AWS_ENDPOINT_URL: {os.environ.get('AWS_ENDPOINT_URL', 'NOT SET')}")

    workspace = "/opt/airflow/dvc-workspace"
    s3_path   = f"s3://mlops-dvc/{researcher_id}/{dataset}"

    logger.info(f"Running: dvc add --external {s3_path}")
    result = subprocess.run(
        ["dvc", "add", "--external", s3_path],
        capture_output=True, text=True, cwd=workspace,
    )
    logger.info(f"stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "dvc add", result.stderr)

    logger.info("dvc_add complete")


def dvc_push(**ctx):
    logger.info("Starting dvc_push")

    workspace = "/opt/airflow/dvc-workspace"

    result = subprocess.run(
        ["dvc", "push"],
        capture_output=True, text=True, cwd=workspace,
    )
    logger.info(f"stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "dvc push", result.stderr)

    logger.info("dvc_push complete")


with DAG(
    dag_id="ingest_dag",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ingestion"],
) as dag:

    dataset       = "{{ dag_run.conf['dataset'] }}"
    researcher_id = "{{ dag_run.conf['researcher_id'] }}"

    t1 = PythonOperator(
        task_id="dvc_add",
        python_callable=dvc_add,
        op_kwargs={"dataset": dataset, "researcher_id": researcher_id},
    )

    t2 = PythonOperator(
        task_id="dvc_push",
        python_callable=dvc_push,
    )

    t1 >> t2