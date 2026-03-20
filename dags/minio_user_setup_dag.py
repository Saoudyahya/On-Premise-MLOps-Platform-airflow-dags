from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import secrets
import psycopg2
import json
import logging

logger = logging.getLogger(__name__)

MINIO_ENDPOINT  = "minio-svc.mlops-minio.svc.cluster.local:9000"
MINIO_ADMIN_KEY = "minio-admin"
MINIO_ADMIN_SEC = "minio-admin"
DVC_BUCKET      = "mlops-dvc"

# Dedicated MinIO PostgreSQL — completely separate from MLflow / JupyterHub DBs
MINIO_DB_DSN = (
    "host=minio-postgresql-svc.mlops-minio.svc.cluster.local "
    "port=5432 "
    "dbname=miniodb "
    "user=minio "
    "password=minio-db-password "
    "sslmode=disable"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _researcher_policy(researcher_id: str) -> str:
    """
    IAM policy that restricts a MinIO user to only their own prefix
    inside the mlops-dvc bucket.

    ListBucket is gated by the s3:prefix condition so a researcher
    can list their own objects but cannot enumerate other prefixes.
    Object-level actions are restricted to arn:…/researcher_id/*.
    """
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid":    "AllowListOwnPrefix",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [f"arn:aws:s3:::{DVC_BUCKET}"],
                "Condition": {
                    "StringLike": {
                        "s3:prefix": [
                            f"{researcher_id}",
                            f"{researcher_id}/*",
                        ]
                    }
                },
            },
            {
                "Sid":    "AllowObjectOpsOwnPrefix",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:GetObjectVersion",
                    "s3:HeadObject",
                ],
                "Resource": [f"arn:aws:s3:::{DVC_BUCKET}/{researcher_id}/*"],
            },
        ],
    }
    return json.dumps(policy)


def _get_stored_secret(researcher_id: str) -> str | None:
    """Return the stored MinIO secret for this researcher, or None."""
    with psycopg2.connect(MINIO_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT minio_secret_key "
                "FROM researcher_credentials "
                "WHERE researcher_id = %s",
                (researcher_id,),
            )
            row = cur.fetchone()
    return row[0] if row else None


def _save_credentials(researcher_id: str, access_key: str, secret_key: str) -> None:
    with psycopg2.connect(MINIO_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO researcher_credentials
                    (researcher_id, minio_access_key, minio_secret_key)
                VALUES (%s, %s, %s)
                ON CONFLICT (researcher_id) DO UPDATE
                    SET minio_access_key = EXCLUDED.minio_access_key,
                        minio_secret_key = EXCLUDED.minio_secret_key,
                        updated_at       = NOW()
                """,
                (researcher_id, access_key, secret_key),
            )
        conn.commit()
    logger.info(f"MinIO credentials persisted for '{researcher_id}' in MinIO DB")


# ── Task ───────────────────────────────────────────────────────────────────────

def setup_minio_user(researcher_id, **ctx):
    """
    Idempotent MinIO onboarding for a researcher.

    1. Creates a MinIO user whose access_key equals the researcher_id.
    2. Creates (or replaces) a canned IAM policy scoped to their prefix.
    3. Attaches the policy to the user.
    4. Stores credentials in the MinIO-dedicated PostgreSQL DB.
    5. Pushes access_key / secret_key / endpoint / bucket to XCom.

    If the user already exists and we have a stored secret we return the
    existing credentials without rotating, so presigned URLs and
    existing notebooks keep working.
    """
    from minio.admin import MinioAdmin

    logger.info(f"=== minio_user_setup | researcher={researcher_id} ===")

    access_key  = researcher_id          # predictable, equals researcher_id
    policy_name = f"researcher-{researcher_id}"

    admin = MinioAdmin(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ADMIN_KEY,
        secret_key=MINIO_ADMIN_SEC,
        secure=False,
    )

    # ── 1. Create or verify user ──────────────────────────────────────────────
    user_exists = False
    try:
        admin.user_info(access_key)
        user_exists = True
        logger.info(f"MinIO user '{access_key}' already exists")
    except Exception:
        logger.info(f"MinIO user '{access_key}' not found — creating")

    if user_exists:
        stored = _get_stored_secret(researcher_id)
        if stored:
            secret_key = stored
            logger.info("Returning stored credentials (no rotation needed)")
        else:
            # User exists in MinIO but has no DB record — rotate once so we
            # have a record we control.
            secret_key = secrets.token_urlsafe(32)
            admin.add_user(access_key, secret_key)
            logger.info("Password rotated (no prior DB record found)")
            _save_credentials(researcher_id, access_key, secret_key)
    else:
        secret_key = secrets.token_urlsafe(32)
        admin.add_user(access_key, secret_key)
        logger.info(f"✅ MinIO user '{access_key}' created")
        _save_credentials(researcher_id, access_key, secret_key)

    # ── 2. Create / update scoped IAM policy ─────────────────────────────────
    policy_json = _researcher_policy(researcher_id)
    try:
        admin.add_canned_policy(policy_name, policy_json)
        logger.info(f"✅ Policy '{policy_name}' created/updated")
    except Exception as exc:
        # MinIO returns an error when you try to overwrite an identical policy;
        # log and continue — the policy already exists.
        logger.warning(f"add_canned_policy warning (may already exist): {exc}")

    # ── 3. Attach policy to user ──────────────────────────────────────────────
    admin.set_policy(policy_name, user=access_key)
    logger.info(f"✅ Policy '{policy_name}' attached to '{access_key}'")

    # ── 4. Push to XCom ───────────────────────────────────────────────────────
    ctx["ti"].xcom_push(key="minio_access_key", value=access_key)
    ctx["ti"].xcom_push(key="minio_secret_key", value=secret_key)
    ctx["ti"].xcom_push(key="minio_endpoint",   value=f"http://{MINIO_ENDPOINT}")
    ctx["ti"].xcom_push(key="minio_bucket",     value=DVC_BUCKET)

    logger.info("=" * 60)
    logger.info(f"  researcher_id  : {researcher_id}")
    logger.info(f"  access_key     : {access_key}")
    logger.info(f"  policy         : {policy_name}")
    logger.info(f"  bucket         : {DVC_BUCKET}")
    logger.info(f"  credentials_db : minio-postgresql-svc.mlops-minio:5432/miniodb")
    logger.info("=" * 60)
    logger.info("=== minio_user_setup DONE ===")


# ── DAG ────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="minio_user_setup_dag",
    description="Provision a scoped MinIO user + IAM policy for a researcher",
    default_args={"owner": "mlops", "retries": 1},
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    tags=["minio", "onboarding"],
) as dag:

    PythonOperator(
        task_id="setup_minio_user",
        python_callable=setup_minio_user,
        op_kwargs={
            "researcher_id": "{{ dag_run.conf['researcher_id'] }}",
        },
    )