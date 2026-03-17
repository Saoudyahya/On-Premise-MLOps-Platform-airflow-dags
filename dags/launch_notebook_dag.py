from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import time
import os
import logging

logger = logging.getLogger(__name__)

JUPYTERHUB_HUB_URL    = "http://hub.mlops-jupyterhub.svc.cluster.local:8081"
JUPYTERHUB_PUBLIC_URL = "http://proxy-public.mlops-jupyterhub.svc.cluster.local:80"
JUPYTERHUB_API        = f"{JUPYTERHUB_HUB_URL}/hub/api"


def get_jupyterhub_token() -> str:
    token = os.environ.get("JUPYTERHUB_ADMIN_TOKEN", "").strip()
    if token:
        return token
    raise RuntimeError("Set JUPYTERHUB_ADMIN_TOKEN env var in your Airflow deployment.")


def spawn_server(username, **ctx):
    logger.info(f"=== TASK: spawn_server | user={username} ===")

    token   = get_jupyterhub_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)

    if r.status_code == 200:
        servers = r.json().get("servers", {})
        for name, server_info in servers.items():
            if server_info.get("ready"):
                logger.info(f"Server '{name}' already running — skipping spawn")
                return

    elif r.status_code == 404:
        logger.info(f"User '{username}' does not exist — creating...")
        r_create = requests.post(
            f"{JUPYTERHUB_API}/users/{username}",
            headers=headers, json={}, timeout=10,
        )
        if r_create.status_code not in (200, 201):
            r_create.raise_for_status()
        logger.info(f"✓ User '{username}' created")
    else:
        r.raise_for_status()

    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/server",
        headers=headers, json={}, timeout=10,
    )
    logger.info(f"Spawn response: {r.status_code}")
    if r.status_code not in (201, 202, 400):
        r.raise_for_status()


def poll_until_ready(username, **ctx):
    logger.info(f"=== TASK: poll_until_ready | user={username} ===")

    admin_token = get_jupyterhub_token()
    headers     = {"Authorization": f"token {admin_token}", "Content-Type": "application/json"}

    # ── 1. Wait for server to be ready ────────────────────────────────────────
    for attempt in range(60):
        logger.info(f"[{attempt + 1}/60] Polling server status for '{username}'...")
        r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
        r.raise_for_status()
        user_data = r.json()

        servers = user_data.get("servers", {})
        if servers:
            for name, server_info in servers.items():
                if server_info.get("ready"):
                    logger.info(f"✓ Server '{name}' is ready")
                    break
            else:
                time.sleep(5)
                continue
            break  # server is ready, exit polling loop

        elif isinstance(user_data.get("server"), str) and user_data["server"]:
            break  # legacy single-server API — ready
        else:
            time.sleep(5)
    else:
        raise TimeoutError(f"Server for '{username}' did not become ready within 5 minutes.")

    # ── 2. Create a scoped user token ─────────────────────────────────────────
    # POST /hub/api/users/{username}/tokens returns a short-lived token that
    # authenticates this specific user — no login screen needed.
    logger.info(f"Creating user token for '{username}'...")
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/tokens",
        headers=headers,
        json={
            "note":       "airflow-launch",
            "expires_in": 86400,   # token valid for 24 h; adjust as needed
        },
        timeout=10,
    )
    r.raise_for_status()
    user_token = r.json()["token"]
    logger.info("✓ User token created")

    # ── 3. Build the direct-access URL ────────────────────────────────────────
    # ?token=  is the standard JupyterHub query param — JupyterHub exchanges it
    # for a session cookie immediately, so the researcher never sees a login page.
    url = f"{JUPYTERHUB_PUBLIC_URL}/user/{username}/lab?token={user_token}"
    logger.info(f"✓ Redirect URL ready → {url}")

    ctx["ti"].xcom_push(key="server_url", value=url)
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