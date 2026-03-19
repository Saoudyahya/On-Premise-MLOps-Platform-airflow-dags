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


def _user_already_exists(response: requests.Response) -> bool:
    """
    MLflow returns 400 (not 409) with error_code=RESOURCE_ALREADY_EXISTS
    when a user already exists. Detect both just in case.
    """
    if response.status_code == 409:
        return True
    if response.status_code == 400:
        try:
            return response.json().get("error_code") == "RESOURCE_ALREADY_EXISTS"
        except Exception:
            return False
    return False


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


def _get_existing_credentials(researcher_id: str) -> str | None:
    """
    Return the stored plaintext password for this researcher, or None
    if no record exists yet.
    """
    with psycopg2.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT mlflow_password FROM researcher_credentials WHERE researcher_id = %s",
                (researcher_id,),
            )
            row = cur.fetchone()
    return row[0] if row else None


def create_mlflow_user(researcher_id, **ctx):
    """
    Provision an MLflow user account for a researcher.

    - If the user does NOT exist → create them with a fresh password.
    - If the user ALREADY exists → return the stored credentials as-is
      (no password rotation, so existing notebooks keep working).

    Credentials are always pushed to XCom so Django never queries the DB.
    """
    logger.info(f"=== create_mlflow_user | researcher={researcher_id} ===")

    username = researcher_id

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

    # ── 2. Try to create the user ─────────────────────────────────────────────
    password = secrets.token_urlsafe(16)

    r = requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/users/create",
        json={"username": username, "password": password},
        auth=(ADMIN_USER, ADMIN_PASS),
        timeout=10,
    )

    if r.status_code == 200:
        # New user — persist the fresh password
        logger.info(f"✅ MLflow user created: {username}")
        _upsert_credentials(researcher_id=researcher_id, username=username, password=password)
        logger.info("✅ Credentials stored in researcher_credentials")

    elif _user_already_exists(r):
        # User already exists — return whatever password we stored last time.
        # Do NOT rotate: existing notebooks/experiments still use the old one.
        logger.info(f"User '{username}' already exists — returning stored credentials")
        existing_password = _get_existing_credentials(researcher_id)

        if existing_password:
            password = existing_password
            logger.info("✅ Returning stored credentials")
        else:
            # Edge case: user exists in MLflow but we have no record in our DB
            # (e.g. manually created). Rotate and store so we have a record.
            logger.warning(
                "User exists in MLflow but no record in researcher_credentials — "
                "rotating password to get a stored copy"
            )
            r2 = requests.patch(
                f"{MLFLOW_URL}/api/2.0/mlflow/users/update-password",
                json={"username": username, "password": password},
                auth=(ADMIN_USER, ADMIN_PASS),
                timeout=10,
            )
            r2.raise_for_status()
            _upsert_credentials(researcher_id=researcher_id, username=username, password=password)
            logger.info("✅ Password rotated and stored")

    else:
        raise RuntimeError(f"MLflow /users/create returned {r.status_code}: {r.text}")

    # ── 3. Push to XCom — Django reads from here, never from the DB ──────────
    ctx["ti"].xcom_push(key="researcher_id",   value=researcher_id)
    ctx["ti"].xcom_push(key="mlflow_username", value=username)
    ctx["ti"].xcom_push(key="mlflow_password", value=password)

    logger.info("=" * 60)
    logger.info(f"  researcher_id : {researcher_id}")
    logger.info(f"  username      : {username}")
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