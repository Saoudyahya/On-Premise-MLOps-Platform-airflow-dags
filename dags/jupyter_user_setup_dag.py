from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import secrets
import bcrypt
import psycopg2
import os
import logging

logger = logging.getLogger(__name__)

JUPYTERHUB_API = "http://hub.mlops-jupyterhub.svc.cluster.local:8081/hub/api"

HUB_DB_DSN = (
    "host=jupyterhub-postgresql-svc.mlops-jupyterhub.svc.cluster.local "
    "port=5432 "
    "dbname=jupyterhub "
    "user=jupyterhub "
    "password=jupyterhub-password "
    "sslmode=disable"
)


def get_admin_token() -> str:
    token = os.environ.get("JUPYTERHUB_ADMIN_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set JUPYTERHUB_ADMIN_TOKEN env var in airflow/values.yaml.")
    return token


def _upsert_native_auth_user(username: str, password: str) -> None:
    """
    Write a bcrypt-hashed password into NativeAuthenticator's users_info table.
    Uses SELECT → UPDATE/INSERT because users_info has no UNIQUE constraint.
    """
    hashed: bytes = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users_info WHERE username = %s", (username,))
            exists = cur.fetchone() is not None

            if exists:
                cur.execute(
                    """
                    UPDATE users_info
                       SET password      = %s,
                           is_authorized = true
                     WHERE username = %s
                    """,
                    (hashed, username),
                )
                logger.info(f"NativeAuth password updated for '{username}'")
            else:
                cur.execute(
                    """
                    INSERT INTO users_info
                        (username, password, is_authorized, login_email_sent, has_2fa)
                    VALUES (%s, %s, true, false, false)
                    """,
                    (username, hashed),
                )
                logger.info(f"NativeAuth user created for '{username}'")
        conn.commit()


def _upsert_researcher_credentials(researcher_id: str, username: str, password: str) -> None:
    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO researcher_credentials
                    (researcher_id, jupyter_username, jupyter_password)
                VALUES (%s, %s, %s)
                ON CONFLICT (researcher_id) DO UPDATE
                    SET jupyter_password = EXCLUDED.jupyter_password,
                        updated_at       = NOW()
                """,
                (researcher_id, username, password),
            )
        conn.commit()
    logger.info(f"Credentials stored in researcher_credentials for '{researcher_id}'")


def setup_user(researcher_id, **ctx):
    """
    Create the JupyterHub user and store credentials.
    Does NOT spawn any server — that's launch_notebook_dag's job.

    Idempotent: if the user already exists in Hub, we still refresh the
    password so the credentials table stays in sync.
    """
    logger.info(f"=== jupyter_user_setup | researcher={researcher_id} ===")

    # username == researcher_id throughout this platform
    username = researcher_id
    password = secrets.token_urlsafe(12)

    # 1. Write password into NativeAuth DB
    _upsert_native_auth_user(username, password)

    # 2. Store plaintext credentials for Django to read
    _upsert_researcher_credentials(researcher_id, username, password)

    # 3. Ensure the Hub knows about this user
    token   = get_admin_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)

    if r.status_code == 404:
        r_create = requests.post(
            f"{JUPYTERHUB_API}/users/{username}",
            headers=headers,
            json={},
            timeout=10,
        )
        if r_create.status_code not in (200, 201):
            r_create.raise_for_status()
        logger.info(f"Hub user '{username}' created")
    elif r.status_code == 200:
        logger.info(f"Hub user '{username}' already exists — credentials refreshed")
    else:
        r.raise_for_status()

    logger.info("=" * 60)
    logger.info(f"  researcher_id : {researcher_id}")
    logger.info(f"  username      : {username}")
    logger.info(f"  password      : {password}")
    logger.info(f"  Fetch via     : GET /api/notebook/credentials/{researcher_id}")
    logger.info("=" * 60)
    logger.info("=== jupyter_user_setup DONE ===")


with DAG(
    dag_id="jupyter_user_setup_dag",
    default_args={"owner": "mlops", "retries": 1},
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["notebook", "onboarding"],
) as dag:

    PythonOperator(
        task_id="setup_user",
        python_callable=setup_user,
        op_kwargs={
            "researcher_id": "{{ dag_run.conf['researcher_id'] }}",
        },
    )
