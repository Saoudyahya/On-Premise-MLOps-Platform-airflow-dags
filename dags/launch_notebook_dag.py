from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import secrets
import bcrypt
import psycopg2
import time
import os
import logging

logger = logging.getLogger(__name__)

# Internal cluster URL — API calls stay inside the cluster
JUPYTERHUB_HUB_URL = "http://hub.mlops-jupyterhub.svc.cluster.local:8081"
JUPYTERHUB_API     = f"{JUPYTERHUB_HUB_URL}/hub/api"

# External URL — what the researcher's browser hits
JUPYTERHUB_PUBLIC_URL = os.environ.get(
    "JUPYTERHUB_PUBLIC_URL",
    "http://proxy-public.mlops-jupyterhub.svc.cluster.local:80",
)

# JupyterHub PostgreSQL DSN — NativeAuthenticator + credentials live here
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
    if token:
        return token
    raise RuntimeError("Set JUPYTERHUB_ADMIN_TOKEN env var in airflow/values.yaml.")


def _upsert_native_auth_user(username: str, password: str) -> None:
    """
    Write a bcrypt-hashed password directly into NativeAuthenticator's
    users_info table. is_authorized=True so the user can log in immediately
    without admin approval.
    """
    hashed: bytes = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users_info
                    (username, password, is_authorized, login_email_sent, has_2fa)
                VALUES (%s, %s, true, false, false)
                ON CONFLICT (username) DO UPDATE
                    SET password      = EXCLUDED.password,
                        is_authorized = true
                """,
                (username, hashed),
            )
        conn.commit()
    logger.info(f"✓ NativeAuth password set for '{username}'")


def _upsert_researcher_credentials(username: str, password: str) -> None:
    """
    Persist the plaintext JupyterHub credentials in researcher_credentials
    so they can be retrieved later (e.g. by an API or another DAG).

    ON CONFLICT rotates the password — re-running the DAG always produces
    fresh credentials rather than silently failing.
    """
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
                (username, username, password),
            )
        conn.commit()
    logger.info(f"✓ Credentials stored in researcher_credentials for '{username}'")


def spawn_server(username, **ctx):
    logger.info(f"=== TASK: spawn_server | user={username} ===")

    # ── 1. Generate password, write to NativeAuth DB, persist plaintext ──────
    password = secrets.token_urlsafe(12)

    _upsert_native_auth_user(username, password)
    _upsert_researcher_credentials(username, password)

    # Push password to XCom so poll_until_ready can surface it in logs
    ctx["ti"].xcom_push(key="password", value=password)

    # ── 2. Ensure user exists in JupyterHub's user table ─────────────────────
    token   = get_admin_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)

    if r.status_code == 200:
        servers = r.json().get("servers", {})
        for name, server_info in servers.items():
            if server_info.get("ready"):
                logger.info(f"Server '{name}' already running — skipping spawn")
                return
        logger.info("User exists but no ready server — spawning")

    elif r.status_code == 404:
        logger.info(f"User '{username}' not in hub DB — creating...")
        r_create = requests.post(
            f"{JUPYTERHUB_API}/users/{username}",
            headers=headers, json={}, timeout=10,
        )
        if r_create.status_code not in (200, 201):
            r_create.raise_for_status()
        logger.info(f"✓ Hub user '{username}' created")

    else:
        r.raise_for_status()

    # ── 3. Spawn the notebook server ─────────────────────────────────────────
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/server",
        headers=headers, json={}, timeout=10,
    )
    logger.info(f"Spawn response: {r.status_code}")
    if r.status_code not in (201, 202, 400):
        r.raise_for_status()

    logger.info("=== spawn_server DONE ===")


def poll_until_ready(username, **ctx):
    logger.info(f"=== TASK: poll_until_ready | user={username} ===")

    token   = get_admin_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    # ── Wait for server to be ready ───────────────────────────────────────────
    for attempt in range(60):
        logger.info(f"[{attempt + 1}/60] Polling '{username}'...")

        r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
        r.raise_for_status()
        user_data = r.json()

        servers = user_data.get("servers", {})
        if servers:
            for name, server_info in servers.items():
                ready   = server_info.get("ready", False)
                pending = server_info.get("pending")
                logger.info(f"  Server '{name}': ready={ready}, pending={pending}")
                if ready:
                    logger.info(f"✓ Server '{name}' is ready")
                    break
            else:
                time.sleep(5)
                continue
            break

        elif isinstance(user_data.get("server"), str) and user_data["server"]:
            logger.info("✓ Server ready (legacy API)")
            break

        else:
            logger.info("  Not ready yet...")
            time.sleep(5)

    else:
        raise TimeoutError(f"Server for '{username}' did not become ready within 5 minutes.")

    # ── Return credentials to caller ─────────────────────────────────────────
    password = ctx["ti"].xcom_pull(task_ids="spawn_server", key="password")

    logger.info("=" * 60)
    logger.info(f"  Login URL : {JUPYTERHUB_PUBLIC_URL}")
    logger.info(f"  Username  : {username}")
    logger.info(f"  Password  : {password}")
    logger.info("  Credentials persisted in researcher_credentials (JupyterHub PostgreSQL)")
    logger.info("=" * 60)

    ctx["ti"].xcom_push(key="server_url", value=JUPYTERHUB_PUBLIC_URL)
    ctx["ti"].xcom_push(key="username",   value=username)
    ctx["ti"].xcom_push(key="password",   value=password)

    logger.info("=== poll_until_ready DONE ===")


with DAG(
    dag_id="launch_notebook_dag",
    default_args={"owner": "mlops", "retries": 1},
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["notebook"],
) as dag:

    username = "{{ dag_run.conf['username'] }}"

    t1 = PythonOperator(task_id="spawn_server",     python_callable=spawn_server,     op_kwargs={"username": username})
    t2 = PythonOperator(task_id="poll_until_ready", python_callable=poll_until_ready, op_kwargs={"username": username})

    t1 >> t2