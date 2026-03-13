from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import time
import os
import logging

logger = logging.getLogger(__name__)

JUPYTERHUB_URL = "http://proxy-public.mlops-jupyterhub.svc.cluster.local:80"
JUPYTERHUB_API = f"{JUPYTERHUB_URL}/hub/api"


def get_jupyterhub_token() -> str:
    """
    Fetch a fresh JupyterHub admin API token.
    First tries JUPYTERHUB_ADMIN_TOKEN env var.
    Falls back to creating one via the hub's internal sqlite DB.
    """
    # ── Option 1: token already set as env var ──────────────────────────────
    token = os.environ.get("JUPYTERHUB_ADMIN_TOKEN", "").strip()
    if token:
        logger.info("✓ Using JUPYTERHUB_ADMIN_TOKEN from environment")
        return token

    # ── Option 2: generate one via JupyterHub token API (no token needed) ──
    # This works because JupyterHub exposes a token endpoint for services
    logger.info("JUPYTERHUB_ADMIN_TOKEN not set — attempting to generate one via hub API")

    hub_token_url = f"{JUPYTERHUB_API}/authorizations/token"
    hub_user      = os.environ.get("JUPYTERHUB_ADMIN_USER",     "admin")
    hub_password  = os.environ.get("JUPYTERHUB_ADMIN_PASSWORD", "")

    if hub_password:
        logger.info(f"Requesting token for user '{hub_user}' via password auth")
        r = requests.post(
            hub_token_url,
            json={"username": hub_user, "password": hub_password},
            timeout=10,
        )
        r.raise_for_status()
        token = r.json().get("token", "")
        if token:
            logger.info("✓ Token obtained via password auth")
            return token

    raise RuntimeError(
        "Could not obtain a JupyterHub token. "
        "Set JUPYTERHUB_ADMIN_TOKEN env var in your Airflow deployment."
    )


def stop_stale_server(username, **ctx):
    logger.info(f"=== TASK: stop_stale_server | user={username} ===")

    token   = get_jupyterhub_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    logger.info(f"Checking if user '{username}' exists in JupyterHub...")
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)

    if r.status_code == 404:
        logger.info(f"User '{username}' does not exist yet — nothing to stop")
        return

    r.raise_for_status()
    user_data = r.json()
    logger.info(f"User found. Raw server data: {user_data.get('servers', {})}")

    servers = user_data.get("servers", {})
    if not servers:
        logger.info("No running servers found — nothing to stop")
        return

    for name, server_info in servers.items():
        ready   = server_info.get("ready", False)
        pending = server_info.get("pending")
        logger.info(f"  Server '{name}': ready={ready}, pending={pending}")

        if ready:
            logger.info(f"  Server '{name}' is healthy — leaving it alone")
        else:
            logger.info(f"  Server '{name}' is stuck (pending={pending}) — stopping it")
            del_r = requests.delete(
                f"{JUPYTERHUB_API}/users/{username}/servers/{name}",
                headers=headers,
                timeout=10,
            )
            logger.info(f"  Delete response: {del_r.status_code}")
            time.sleep(3)

    logger.info("=== stop_stale_server DONE ===")


def spawn_server(username, **ctx):
    logger.info(f"=== TASK: spawn_server | user={username} ===")

    token   = get_jupyterhub_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    # Check if already running
    logger.info(f"Checking current server state for '{username}'...")
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)

    if r.status_code == 200:
        servers = r.json().get("servers", {})
        for name, server_info in servers.items():
            if server_info.get("ready"):
                logger.info(f"Server '{name}' already running and ready — skipping spawn")
                return
        logger.info(f"User exists but no ready server found — spawning now")
    elif r.status_code == 404:
        logger.info(f"User '{username}' does not exist — JupyterHub will create it on spawn")
    else:
        r.raise_for_status()

    logger.info(f"Sending spawn request for '{username}'...")
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/server",
        headers=headers,
        json={},
        timeout=10,
    )
    logger.info(f"Spawn response: {r.status_code}")

    if r.status_code == 201:
        logger.info("Server created immediately (201)")
    elif r.status_code == 202:
        logger.info("Spawn accepted, server is starting (202)")
    else:
        logger.error(f"Unexpected response: {r.status_code} — {r.text}")
        r.raise_for_status()

    logger.info("=== spawn_server DONE ===")


def poll_until_ready(username, **ctx):
    logger.info(f"=== TASK: poll_until_ready | user={username} ===")

    token   = get_jupyterhub_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    for attempt in range(60):
        logger.info(f"[{attempt + 1}/60] Polling server status for '{username}'...")

        r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
        r.raise_for_status()
        user_data = r.json()

        # JupyterHub ≥2 — servers dict
        servers = user_data.get("servers", {})
        if servers:
            for name, server_info in servers.items():
                ready   = server_info.get("ready", False)
                pending = server_info.get("pending")
                logger.info(f"  Server '{name}': ready={ready}, pending={pending}")
                if ready:
                    url = f"{JUPYTERHUB_URL}/user/{username}/lab"
                    logger.info(f"✓ Server is ready → {url}")
                    ctx["ti"].xcom_push(key="server_url", value=url)
                    logger.info("=== poll_until_ready DONE ===")
                    return

        # Legacy single-server API
        elif isinstance(user_data.get("server"), str) and user_data["server"]:
            url = f"{JUPYTERHUB_URL}/user/{username}/lab"
            logger.info(f"✓ Server ready (legacy API) → {url}")
            ctx["ti"].xcom_push(key="server_url", value=url)
            logger.info("=== poll_until_ready DONE ===")
            return
        else:
            logger.info("  No server entry yet — still starting...")

        time.sleep(5)

    raise TimeoutError(f"Server for '{username}' did not become ready within 5 minutes.")


with DAG(
    dag_id="launch_notebook_dag",
    default_args={"owner": "mlops", "retries": 1},
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["notebook"],
) as dag:

    username = "{{ dag_run.conf['username'] }}"

    t1 = PythonOperator(task_id="stop_stale_server", python_callable=stop_stale_server, op_kwargs={"username": username})
    t2 = PythonOperator(task_id="spawn_server",      python_callable=spawn_server,      op_kwargs={"username": username})
    t3 = PythonOperator(task_id="poll_until_ready",  python_callable=poll_until_ready,  op_kwargs={"username": username})

    t1 >> t2 >> t3