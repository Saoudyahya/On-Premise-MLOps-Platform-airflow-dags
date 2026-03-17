from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import time
import os
import logging

logger = logging.getLogger(__name__)

# Internal cluster URL — used for API calls from within the cluster
JUPYTERHUB_HUB_URL = "http://hub.mlops-jupyterhub.svc.cluster.local:8081"
JUPYTERHUB_API     = f"{JUPYTERHUB_HUB_URL}/hub/api"

# External URL — handed back to the researcher's browser
JUPYTERHUB_PUBLIC_URL = os.environ.get(
    "JUPYTERHUB_PUBLIC_URL",
    "http://proxy-public.mlops-jupyterhub.svc.cluster.local:80",  # fallback
)


def get_jupyterhub_token() -> str:
    token = os.environ.get("JUPYTERHUB_ADMIN_TOKEN", "").strip()
    if token:
        return token
    raise RuntimeError(
        "Could not obtain a JupyterHub token. "
        "Set JUPYTERHUB_ADMIN_TOKEN env var in your Airflow deployment."
    )


def spawn_server(username, **ctx):
    logger.info(f"=== TASK: spawn_server | user={username} ===")

    token   = get_jupyterhub_token()
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    # ── Check if user + server already exist ─────────────────────────────────
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)

    if r.status_code == 200:
        servers = r.json().get("servers", {})
        for name, server_info in servers.items():
            if server_info.get("ready"):
                logger.info(f"Server '{name}' already running and ready — skipping spawn")
                return
        logger.info("User exists but no ready server — proceeding to spawn")

    elif r.status_code == 404:
        logger.info(f"User '{username}' does not exist — creating first...")
        r_create = requests.post(
            f"{JUPYTERHUB_API}/users/{username}",
            headers=headers, json={}, timeout=10,
        )
        logger.info(f"User creation response: {r_create.status_code}")
        if r_create.status_code not in (200, 201):
            r_create.raise_for_status()
        logger.info(f"✓ User '{username}' created")

    else:
        r.raise_for_status()

    # ── Spawn ─────────────────────────────────────────────────────────────────
    logger.info(f"Sending spawn request for '{username}'...")
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/server",
        headers=headers, json={}, timeout=10,
    )
    logger.info(f"Spawn response: {r.status_code}")

    if r.status_code == 201:
        logger.info("Server created immediately (201)")
    elif r.status_code == 202:
        logger.info("Spawn accepted, server is starting (202)")
    elif r.status_code == 400:
        logger.info(f"Server already exists or is spawning (400) — {r.text}")
    else:
        r.raise_for_status()

    logger.info("=== spawn_server DONE ===")


def poll_until_ready(username, **ctx):
    logger.info(f"=== TASK: poll_until_ready | user={username} ===")

    admin_token = get_jupyterhub_token()
    headers     = {"Authorization": f"token {admin_token}", "Content-Type": "application/json"}

    # ── 1. Wait for server to be ready ───────────────────────────────────────
    for attempt in range(60):
        logger.info(f"[{attempt + 1}/60] Polling server status for '{username}'...")

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
            logger.info("  No server entry yet — still starting...")
            time.sleep(5)

    else:
        raise TimeoutError(f"Server for '{username}' did not become ready within 5 minutes.")

    # ── 2. Create a scoped user token ────────────────────────────────────────
    logger.info(f"Creating user token for '{username}'...")
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/tokens",
        headers=headers,
        json={
            "note":       "airflow-launch",
            "expires_in": 86400,  # 24 h — shorten to e.g. 300 for one-time links
        },
        timeout=10,
    )
    r.raise_for_status()
    user_token = r.json()["token"]
    logger.info("✓ User token created")

    # ── 3. Build redirect URL ─────────────────────────────────────────────────
    # Token goes to the SINGLEUSER server, not /hub/login.
    # The singleuser server validates it against the Hub via HubOAuth.
    # This requires `c.HubOAuth.allow_token_in_url = True` in
    # singleuser.extraConfig in jupyterhub/values.yaml.
    url = f"{JUPYTERHUB_PUBLIC_URL}/user/{username}/lab?token={user_token}"
    logger.info(f"✓ Redirect URL → {url}")

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