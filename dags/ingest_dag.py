from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import boto3
import subprocess
import os
import logging

logger = logging.getLogger(__name__)
default_args = {"owner": "mlops", "retries": 0}


def dvc_add(dataset, researcher_id, **ctx):
    logger.info("=" * 50)
    logger.info(f"Starting dvc_add for {researcher_id}/{dataset}")
    logger.info(f"AWS_ENDPOINT_URL: {os.environ.get('AWS_ENDPOINT_URL', 'NOT SET')}")
    logger.info(f"AWS_ACCESS_KEY_ID: {os.environ.get('AWS_ACCESS_KEY_ID', 'NOT SET')}")

    workspace = "/opt/airflow/dvc-workspace"
    s3_path   = f"s3://mlops-dvc/{researcher_id}/{dataset}"

    # Check workspace exists
    if not os.path.exists(workspace):
        logger.error(f"DVC workspace not found at {workspace}")
        raise FileNotFoundError(f"DVC workspace not found: {workspace}")
    logger.info(f"Workspace found: {workspace}")

    # Check DVC is installed
    result = subprocess.run(["which", "dvc"], capture_output=True, text=True)
    logger.info(f"DVC path: {result.stdout.strip() or 'NOT FOUND'}")

    # Check DVC remote config
    result = subprocess.run(
        ["dvc", "remote", "list"],
        capture_output=True, text=True, cwd=workspace
    )
    logger.info(f"DVC remotes: {result.stdout.strip() or 'NONE CONFIGURED'}")
    if result.stderr:
        logger.warning(f"DVC remote list stderr: {result.stderr}")

    # Check file exists in MinIO before tracking
    logger.info(f"Checking MinIO for: {s3_path}")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["AWS_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        s3.head_object(Bucket="mlops-dvc", Key=f"{researcher_id}/{dataset}")
        logger.info(f"File found in MinIO: mlops-dvc/{researcher_id}/{dataset}")
    except Exception as e:
        logger.error(f"File NOT found in MinIO: {e}")
        raise

    # Run dvc add --external
    logger.info(f"Running: dvc add --external {s3_path}")
    result = subprocess.run(
        ["dvc", "add", "--external", s3_path],
        capture_output=True, text=True, cwd=workspace
    )
    logger.info(f"dvc add stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"dvc add stderr: {result.stderr}")
    if result.returncode != 0:
        logger.error(f"dvc add failed with exit code {result.returncode}")
        raise subprocess.CalledProcessError(result.returncode, "dvc add", result.stderr)

    dvc_file = f"{researcher_id}/{dataset}.dvc"
    logger.info(f"DVC tracking complete: {dvc_file}")
    ctx["ti"].xcom_push(key="dvc_file", value=dvc_file)


def git_commit(dataset, researcher_id, **ctx):
    logger.info("=" * 50)
    logger.info(f"Starting git_commit for {researcher_id}/{dataset}")

    workspace = "/opt/airflow/dvc-workspace"
    dvc_file  = ctx["ti"].xcom_pull(key="dvc_file")
    gitignore = os.path.join(researcher_id, ".gitignore")

    logger.info(f"DVC file to commit: {dvc_file}")

    # Check git status
    result = subprocess.run(
        ["git", "status"], capture_output=True, text=True, cwd=workspace
    )
    logger.info(f"Git status:\n{result.stdout}")

    for cmd in [
        ["git", "config", "user.email", "airflow@mlops"],
        ["git", "config", "user.name", "Airflow"],
    ]:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=workspace)
        logger.info(f"Ran: {' '.join(cmd)} → rc={result.returncode}")

    result = subprocess.run(
        ["git", "add", dvc_file, gitignore],
        capture_output=True, text=True, cwd=workspace
    )
    logger.info(f"git add → rc={result.returncode} stdout={result.stdout} stderr={result.stderr}")

    result = subprocess.run(
        ["git", "commit", "-m", f"[ingest] {researcher_id}/{dataset}"],
        capture_output=True, text=True, cwd=workspace
    )
    logger.info(f"git commit stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"git commit stderr: {result.stderr}")
    if result.returncode not in (0, 1):
        logger.error(f"git commit failed: rc={result.returncode}")
        raise subprocess.CalledProcessError(result.returncode, "git commit", result.stderr)

    logger.info("git_commit complete")


def git_push(**ctx):
    logger.info("=" * 50)
    logger.info("Starting git_push")

    workspace = "/opt/airflow/dvc-workspace"

    result = subprocess.run(
        ["git", "push"],
        capture_output=True, text=True, cwd=workspace
    )
    logger.info(f"git push stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"git push stderr: {result.stderr}")
    if result.returncode != 0:
        logger.error(f"git push failed: rc={result.returncode}")
        raise subprocess.CalledProcessError(result.returncode, "git push", result.stderr)

    logger.info("git_push complete")


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
        task_id="git_commit",
        python_callable=git_commit,
        op_kwargs={"dataset": dataset, "researcher_id": researcher_id},
    )

    t3 = PythonOperator(
        task_id="git_push",
        python_callable=git_push,
    )

    t1 >> t2 >> t3