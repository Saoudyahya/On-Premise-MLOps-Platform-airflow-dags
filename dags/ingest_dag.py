from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import boto3
import subprocess
import os

default_args = {"owner": "mlops", "retries": 1}

def stage_raw_file(dataset, **ctx):
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3.upload_file(f"/data/incoming/{dataset}", "mlops-raw", dataset)

def dvc_add(dataset, **ctx):
    subprocess.run(
        ["dvc", "add", f"data/{dataset}"],
        check=True,
        cwd="/opt/airflow/dvc-workspace"
    )

def dvc_push(**ctx):
    subprocess.run(
        ["dvc", "push"],
        check=True,
        cwd="/opt/airflow/dvc-workspace"
    )

with DAG(
    dag_id="ingest_dag",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,              # ← was schedule_interval
    catchup=False,
    tags=["ingestion"],
) as dag:

    dataset = "{{ dag_run.conf['dataset'] }}"

    t1 = PythonOperator(
        task_id="stage_raw_file",
        python_callable=stage_raw_file,
        op_kwargs={"dataset": dataset},
    )

    t2 = PythonOperator(
        task_id="dvc_add",
        python_callable=dvc_add,
        op_kwargs={"dataset": dataset},
    )

    t3 = PythonOperator(
        task_id="dvc_push",
        python_callable=dvc_push,
    )

    t1 >> t2 >> t3