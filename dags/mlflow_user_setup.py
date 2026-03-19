from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import secrets
import psycopg2
import requests
import logging

logger = logging.getLogger(__name__)

MLFLOW_URL = "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000"
ADMIN_USER = "admin"
ADMIN_PASS = "mlops-admin-2024"   # must match auth.ini → admin_password

DB_DSN = (
    "host=mlflow-postgresql-svc.mlops-mlflow.svc.cluster.local "
    "port=5432 "
    "dbname=mlflow "
    "user=mlflow "
    "password=mlflow-password "
    "sslmode=disable"
)


def _upsert_credentials(researcher_id: str, username: str, password: str) -> None:
    with psycopg2.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO researcher_credentials
                    (researcher_id, mlflow_username, mlflow_password)
                VALUES (%s, %s, %s)
                ON CONFLICT (researcher_id) DO UPDATE
                    SET mlflow_password = EXCLUDED.mlflow_password,
                        updated_at      = NOW()
                """,
                (researcher_id, username, password),
            )
        conn.commit()


def create_mlflow_user(researcher_id, **ctx):
    """
    Provision an MLflow user account for a researcher.

    Steps:
      1. Verify admin credentials work against the MLflow server.
      2. Create the user (or rotate the password if they already exist).
      3. Persist credentials in researcher_credentials (MLflow PostgreSQL).
      4. Push credentials to XCom so Django can return them without a DB call.

    Idempotent: safe to re-run; existing users get a fresh password.
    """
    logger.info(f"=== create_mlflow_user | researcher={researcher_id} ===")

    username = researcher_id
    password = secrets.token_urlsafe(16)

    # ── 1. Verify admin credentials ───────────────────────────────────────────
    probe = requests.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/users/get",
        params={"username": ADMIN_USER},
        auth=(ADMIN_USER, ADMIN_PASS),
        timeout=10,
    )
    if probe.status_code == 403:
        raise RuntimeError(
            "Admin credentials rejected by MLflow (403). "
            "Verify ADMIN_PASS matches auth.ini → admin_password on the MLflow pod."
        )
    logger.info(f"Admin auth probe: HTTP {probe.status_code}")

    # ── 2. Create user or rotate password ─────────────────────────────────────
    r = requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/users/create",
        json={"username": username, "password": password},
        auth=(ADMIN_USER, ADMIN_PASS),
        timeout=10,
    )

    if r.status_code == 200:
        logger.info(f"✅ MLflow user created: {username}")

    elif r.status_code == 409:
        logger.info(f"User '{username}' already exists — rotating password")
        r2 = requests.patch(
            f"{MLFLOW_URL}/api/2.0/mlflow/users/update-password",
            json={"username": username, "password": password},
            auth=(ADMIN_USER, ADMIN_PASS),
            timeout=10,
        )
        r2.raise_for_status()
        logger.info(f"✅ Password rotated for: {username}")

    else:
        raise RuntimeError(
            f"MLflow /users/create returned {r.status_code}: {r.text}"
        )

    # ── 3. Persist in PostgreSQL ───────────────────────────────────────────────
    _upsert_credentials(researcher_id=researcher_id, username=username, password=password)
    logger.info("✅ Credentials stored in researcher_credentials (mlflow PostgreSQL)")

    # ── 4. Push to XCom so Django never needs to query the DB ─────────────────
    ctx["ti"].xcom_push(key="researcher_id",   value=researcher_id)
    ctx["ti"].xcom_push(key="mlflow_username", value=username)
    ctx["ti"].xcom_push(key="mlflow_password", value=password)

    logger.info("=" * 60)
    logger.info(f"  researcher_id : {researcher_id}")
    logger.info(f"  username      : {username}")
    logger.info(f"  password      : {password}")
    logger.info("=" * 60)
    logger.info("=== create_mlflow_user DONE ===")


with DAG(
    dag_id="mlflow_user_setup",
    description="Provision an MLflow user account for a researcher",
    default_args={"owner": "mlops", "retries": 1},
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    tags=["mlflow", "onboarding"],
) as dag:

    PythonOperator(
        task_id="create_mlflow_user",
        python_callable=create_mlflow_user,
        op_kwargs={
            "researcher_id": "{{ dag_run.conf['researcher_id'] }}",
        },
    )