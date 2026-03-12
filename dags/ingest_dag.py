from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timezone
import subprocess
import os
import json
import logging
import boto3

logger = logging.getLogger(__name__)

WORKSPACE    = "/opt/airflow/dvc-workspace"
MINIO_URL    = os.environ.get("AWS_ENDPOINT_URL", "http://minio-svc.mlops-minio.svc.cluster.local:9000")
MINIO_KEY    = os.environ.get("AWS_ACCESS_KEY_ID", "minio-admin")
MINIO_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio-admin")


def ingest_and_version(dataset, researcher_id, **ctx):
    # ── 1. Setup workspace ──────────────────────────────────────────
    os.makedirs(WORKSPACE, exist_ok=True)
    logger.info(f"Workspace ready: {WORKSPACE}")

    def run(cmd, **kwargs):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE, **kwargs)
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
        return result

    # Git init
    if not os.path.exists(os.path.join(WORKSPACE, ".git")):
        run(["git", "init", "-b", "main"])
        run(["git", "config", "user.email", "airflow@mlops"])
        run(["git", "config", "user.name",  "Airflow"])
        logger.info("Git initialized")

    # DVC init
    if not os.path.exists(os.path.join(WORKSPACE, ".dvc")):
        run(["dvc", "init"])
        logger.info("DVC initialized")

    # Configure MinIO remote
    remote_check = subprocess.run(
        ["dvc", "remote", "list"], capture_output=True, text=True, cwd=WORKSPACE
    )
    if "mlops-minio" not in remote_check.stdout:
        run(["dvc", "remote", "add", "-d", "mlops-minio", "s3://mlops-dvc"])
        run(["dvc", "remote", "modify", "mlops-minio", "endpointurl", MINIO_URL])
        run(["dvc", "remote", "modify", "mlops-minio", "access_key_id", MINIO_KEY])
        run(["dvc", "remote", "modify", "mlops-minio", "secret_access_key", MINIO_SECRET])
        logger.info("DVC remote configured → MinIO")

    # ── 2. DVC add ──────────────────────────────────────────────────
    s3_path = f"s3://mlops-dvc/{researcher_id}/{dataset}"
    logger.info(f"dvc add --external {s3_path}")
    run(["dvc", "add", "--external", s3_path])

    # ── 3. DVC push ─────────────────────────────────────────────────
    logger.info("dvc push")
    run(["dvc", "push"])

    # ── 4. Register version in MinIO ────────────────────────────────
    dvc_file = os.path.join(WORKSPACE, f"{dataset}.dvc")
    version_hash = "unknown"
    if os.path.exists(dvc_file):
        with open(dvc_file) as f:
            for line in f:
                if "md5:" in line:
                    version_hash = line.split("md5:")[-1].strip()
                    break

    version_info = {
        "dataset":       dataset,
        "researcher_id": researcher_id,
        "version_hash":  version_hash,
        "s3_path":       s3_path,
        "ingested_at":   datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"Version: {json.dumps(version_info)}")

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_KEY,
        aws_secret_access_key=MINIO_SECRET,
    )

    versions_key = f"{researcher_id}/.versions/{dataset}.json"
    try:
        existing = s3.get_object(Bucket="mlops-dvc", Key=versions_key)
        versions = json.loads(existing["Body"].read())
    except Exception:
        versions = []

    versions.append(version_info)
    s3.put_object(
        Bucket="mlops-dvc",
        Key=versions_key,
        Body=json.dumps(versions, indent=2),
        ContentType="application/json",
    )
    logger.info(f"Saved {len(versions)} version(s) to s3://mlops-dvc/{versions_key}")


with DAG(
    dag_id="ingest_dag",
    default_args={"owner": "mlops", "retries": 0},
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ingestion"],
) as dag:

    PythonOperator(
        task_id="ingest_and_version",
        python_callable=ingest_and_version,
        op_kwargs={
            "dataset":       "{{ dag_run.conf['dataset'] }}",
            "researcher_id": "{{ dag_run.conf['researcher_id'] }}",
        },
    )