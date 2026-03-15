from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import time
import os
import logging

logger = logging.getLogger(__name__)

# ── FIX 1: Use the hub service directly, NOT proxy-public ──────────────────────
# proxy-public:80  → user-facing CHP proxy  (for browser traffic)
# hub:8081         → Hub API               (for programmatic API calls)
# Calling /hub/api through proxy-public fails with 404 when the user doesn't
# exist yet because CHP has no route registered for that user.
JUPYTERHUB_HUB_URL    = "http://hub.mlops-jupyterhub.svc.cluster.local:8081"
JUPYTERHUB_PUBLIC_URL = "http://proxy-public.mlops-jupyterhub.svc.cluster.local:80"
JUPYTERHUB_API        = f"{JUPYTERHUB_HUB_URL}/hub/api"


def get_jupyterhub_token() -> str:
    token = os.environ.get("JUPYTERHUB_ADMIN_TOKEN", "").strip()
    if token:
        logger.info("✓ Using JUPYTERHUB_ADMIN_TOKEN from environment")
        return token
    raise RuntimeError(
        "Could not obtain a JupyterHub token. "
        "Set JUPYTERHUB_ADMIN_TOKEN env var in your Airflow deployment."
    )


def spawn_server(username, **ctx):
    logger.info(f"=== TASK: spawn_server | user={username} ===")

    token   = get_jupyterhub_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    # ── Step 1: Check if user exists ────────────────────────────────────────────
    logger.info(f"Checking current server state for '{username}'...")
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)

    if r.status_code == 200:
        servers = r.json().get("servers", {})
        for name, server_info in servers.items():
            if server_info.get("ready"):
                logger.info(f"Server '{name}' already running and ready — skipping spawn")
                return
        logger.info("User exists but no ready server found — proceeding to spawn")

    elif r.status_code == 404:
        # ── FIX 2: Create user first before spawning ─────────────────────────────
        # JupyterHub requires the user to exist in its DB before a server can be
        # spawned via POST /hub/api/users/{username}/server.
        # POST /hub/api/users/{username}/server on a non-existent user → 404.
        logger.info(f"User '{username}' does not exist — creating user first...")
        r_create = requests.post(
            f"{JUPYTERHUB_API}/users/{username}",
            headers=headers,
            json={},
            timeout=10,
        )
        logger.info(f"User creation response: {r_create.status_code}")
        if r_create.status_code not in (200, 201):
            logger.error(f"Failed to create user: {r_create.status_code} — {r_create.text}")
            r_create.raise_for_status()
        logger.info(f"✓ User '{username}' created")

    else:
        r.raise_for_status()

    # ── Step 2: Spawn the server ────────────────────────────────────────────────
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
    elif r.status_code == 400:
        # Server already exists / is already spawning
        logger.info(f"Server already exists or is spawning (400) — {r.text}")
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
                    # Use proxy-public URL for the final user-facing link
                    url = f"{JUPYTERHUB_PUBLIC_URL}/user/{username}/lab"
                    logger.info(f"✓ Server is ready → {url}")
                    ctx["ti"].xcom_push(key="server_url", value=url)
                    logger.info("=== poll_until_ready DONE ===")
                    return

        # Legacy single-server API
        elif isinstance(user_data.get("server"), str) and user_data["server"]:
            url = f"{JUPYTERHUB_PUBLIC_URL}/user/{username}/lab"
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
    is_paused_upon_creation=False,   # ← add this
    tags=["notebook"],
) as dag:

    username = "{{ dag_run.conf['username'] }}"

    t1 = PythonOperator(task_id="spawn_server",     python_callable=spawn_server,     op_kwargs={"username": username})
    t2 = PythonOperator(task_id="poll_until_ready", python_callable=poll_until_ready, op_kwargs={"username": username})

    t1 >> t2