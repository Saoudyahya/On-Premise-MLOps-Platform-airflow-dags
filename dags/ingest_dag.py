from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os
import json
import logging

logger = logging.getLogger(__name__)
default_args = {"owner": "mlops", "retries": 0}

WORKSPACE   = "/opt/airflow/dvc-workspace"
MINIO_URL   = os.environ.get("AWS_ENDPOINT_URL", "http://minio-svc.mlops-minio.svc.cluster.local:9000")
MINIO_KEY   = os.environ.get("AWS_ACCESS_KEY_ID", "minio-admin")
MINIO_SECRET= os.environ.get("AWS_SECRET_ACCESS_KEY", "minio-admin")


def setup_workspace(**ctx):
    """
    Create and initialize DVC workspace if it doesn't exist.
    Configure MinIO as the DVC remote.
    """
    logger.info("Setting up DVC workspace")

    # Create workspace dir
    os.makedirs(WORKSPACE, exist_ok=True)
    logger.info(f"Workspace dir ready: {WORKSPACE}")

    # Init git if not already
    if not os.path.exists(os.path.join(WORKSPACE, ".git")):
        subprocess.run(["git", "init"], cwd=WORKSPACE, check=True)
        subprocess.run(["git", "config", "user.email", "airflow@mlops"], cwd=WORKSPACE, check=True)
        subprocess.run(["git", "config", "user.name", "Airflow"], cwd=WORKSPACE, check=True)
        logger.info("Git initialized")

    # Init DVC if not already
    if not os.path.exists(os.path.join(WORKSPACE, ".dvc")):
        subprocess.run(["dvc", "init"], cwd=WORKSPACE, check=True)
        logger.info("DVC initialized")

    # Configure MinIO as DVC remote
    result = subprocess.run(
        ["dvc", "remote", "list"],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    if "mlops-minio" not in result.stdout:
        subprocess.run([
            "dvc", "remote", "add", "-d", "mlops-minio",
            "s3://mlops-dvc"
        ], cwd=WORKSPACE, check=True)

        subprocess.run([
            "dvc", "remote", "modify", "mlops-minio",
            "endpointurl", MINIO_URL
        ], cwd=WORKSPACE, check=True)

        subprocess.run([
            "dvc", "remote", "modify", "mlops-minio",
            "access_key_id", MINIO_KEY
        ], cwd=WORKSPACE, check=True)

        subprocess.run([
            "dvc", "remote", "modify", "mlops-minio",
            "secret_access_key", MINIO_SECRET
        ], cwd=WORKSPACE, check=True)

        logger.info("DVC remote configured → MinIO")

    # List remotes for confirmation
    result = subprocess.run(
        ["dvc", "remote", "list"],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    logger.info(f"DVC remotes: {result.stdout}")


def dvc_add(dataset, researcher_id, **ctx):
    """
    Track the file already in MinIO using DVC external tracking.
    Creates a versioned .dvc pointer file per researcher/dataset.
    """
    logger.info(f"dvc_add: {researcher_id}/{dataset}")

    s3_path      = f"s3://mlops-dvc/{researcher_id}/{dataset}"
    dataset_dir  = os.path.join(WORKSPACE, researcher_id)
    os.makedirs(dataset_dir, exist_ok=True)

    logger.info(f"Tracking external file: {s3_path}")
    result = subprocess.run(
        ["dvc", "add", "--external", s3_path],
        capture_output=True, text=True, cwd=WORKSPACE,
    )
    logger.info(f"stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "dvc add", result.stderr)

    # Read the generated .dvc file to get the version hash
    dvc_file_path = os.path.join(WORKSPACE, f"{researcher_id}/{dataset}.dvc")

    # DVC puts it relative to cwd — check both locations
    alt_path = os.path.join(WORKSPACE, f"{dataset}.dvc")
    if os.path.exists(alt_path):
        dvc_file_path = alt_path

    if os.path.exists(dvc_file_path):
        with open(dvc_file_path) as f:
            dvc_meta = f.read()
        logger.info(f".dvc file contents:\n{dvc_meta}")
        ctx["ti"].xcom_push(key="dvc_meta", value=dvc_meta)

    ctx["ti"].xcom_push(key="dvc_file", value=f"{researcher_id}/{dataset}.dvc")
    ctx["ti"].xcom_push(key="s3_path", value=s3_path)
    logger.info("dvc_add complete")


def dvc_push(**ctx):
    """Push DVC cache to MinIO remote."""
    logger.info("Starting dvc_push")

    result = subprocess.run(
        ["dvc", "push"],
        capture_output=True, text=True, cwd=WORKSPACE,
    )
    logger.info(f"stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "dvc push", result.stderr)

    logger.info("dvc_push complete")


def register_version(dataset, researcher_id, **ctx):
    """
    Save version metadata to MinIO so the API can list versions per researcher.
    Stored at: s3://mlops-dvc/<researcher_id>/.versions/<dataset>.json
    """
    import boto3
    import hashlib

    logger.info(f"Registering version for {researcher_id}/{dataset}")

    dvc_meta = ctx["ti"].xcom_pull(key="dvc_meta") or ""

    # Extract md5 hash from .dvc file
    version_hash = "unknown"
    for line in dvc_meta.splitlines():
        if "md5:" in line:
            version_hash = line.split("md5:")[-1].strip()
            break

    version_info = {
        "dataset":       dataset,
        "researcher_id": researcher_id,
        "version_hash":  version_hash,
        "s3_path":       f"s3://mlops-dvc/{researcher_id}/{dataset}",
        "ingested_at":   datetime.utcnow().isoformat() + "Z",
    }

    logger.info(f"Version info: {json.dumps(version_info, indent=2)}")

    # Store version metadata in MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_KEY,
        aws_secret_access_key=MINIO_SECRET,
    )

    versions_key = f"{researcher_id}/.versions/{dataset}.json"

    # Load existing versions if any
    try:
        existing = s3.get_object(Bucket="mlops-dvc", Key=versions_key)
        versions = json.loads(existing["Body"].read())
        if not isinstance(versions, list):
            versions = []
    except Exception:
        versions = []

    versions.append(version_info)

    s3.put_object(
        Bucket="mlops-dvc",
        Key=versions_key,
        Body=json.dumps(versions, indent=2),
        ContentType="application/json",
    )

    logger.info(f"Version saved to s3://mlops-dvc/{versions_key}")
    logger.info(f"Total versions for {dataset}: {len(versions)}")


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

    t0 = PythonOperator(
        task_id="setup_workspace",
        python_callable=setup_workspace,
    )

    t1 = PythonOperator(
        task_id="dvc_add",
        python_callable=dvc_add,
        op_kwargs={"dataset": dataset, "researcher_id": researcher_id},
    )

    t2 = PythonOperator(
        task_id="dvc_push",
        python_callable=dvc_push,
    )

    t3 = PythonOperator(
        task_id="register_version",
        python_callable=register_version,
        op_kwargs={"dataset": dataset, "researcher_id": researcher_id},
    )

    t0 >> t1 >> t2 >> t3