from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import time
import os

default_args = {"owner": "mlops", "retries": 1}

JUPYTERHUB_URL = "http://proxy-public.mlops-jupyterhub.svc.cluster.local:80"
JUPYTERHUB_API = f"{JUPYTERHUB_URL}/hub/api"
ADMIN_TOKEN    = os.environ.get("JUPYTERHUB_ADMIN_TOKEN", "")

HEADERS = {
    "Authorization": f"token {ADMIN_TOKEN}",
    "Content-Type":  "application/json",
}


def check_user_exists(username, **ctx):
    """Create the JupyterHub user if they don't exist yet."""
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS)
    if r.status_code == 404:
        requests.post(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS).raise_for_status()
        print(f"Created JupyterHub user: {username}")
    elif r.status_code == 200:
        print(f"User already exists: {username}")
    else:
        r.raise_for_status()


def stop_stale_server(username, **ctx):
    """
    Stop the server only if it is stuck in a pending/unknown state.
    A healthy running server is left alone so the researcher doesn't lose work.
    """
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS)
    r.raise_for_status()
    user_data = r.json()

    # JupyterHub ≥2 uses "servers" dict; fallback to legacy "server" string
    servers = user_data.get("servers", {})
    if not servers and user_data.get("server"):
        # Legacy single-server API — server is a path string when running
        print("Server already running (legacy API), leaving it alone.")
        return

    for name, server_info in servers.items():
        ready   = server_info.get("ready", False)
        pending = server_info.get("pending")          # "spawn" | "stop" | None

        if pending == "spawn":
            print(f"Server '{name}' stuck in pending spawn — stopping it.")
            requests.delete(
                f"{JUPYTERHUB_API}/users/{username}/servers/{name}",
                headers=HEADERS
            )
            time.sleep(3)
        elif ready:
            print(f"Server '{name}' already running and healthy — skipping stop.")
        else:
            print(f"Server '{name}' in unknown state (pending={pending}) — stopping it.")
            requests.delete(
                f"{JUPYTERHUB_API}/users/{username}/servers/{name}",
                headers=HEADERS
            )
            time.sleep(3)


def spawn_server(username, profile, **ctx):
    """
    Spawn a new named server. If a healthy server is already running, skip.
    """
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS)
    r.raise_for_status()
    user_data = r.json()

    # Check if a ready server already exists
    servers = user_data.get("servers", {})
    for name, server_info in servers.items():
        if server_info.get("ready"):
            print(f"Server '{name}' already ready — no need to spawn.")
            ctx["ti"].xcom_push(key="server_url", value=f"{JUPYTERHUB_URL}/user/{username}/lab")
            return

    # Spawn with the requested profile
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/server",
        headers=HEADERS,
        json={"profile_name": profile},
    )
    if r.status_code not in (201, 202):
        r.raise_for_status()
    print(f"Spawn requested for {username} with profile: {profile}")


def poll_until_ready(username, **ctx):
    """
    Poll every 5 s up to 5 min until the server is ready.
    Works with both JupyterHub ≥2 (servers dict) and legacy (server string).
    """
    for attempt in range(60):
        r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS)
        r.raise_for_status()
        user_data = r.json()

        # JupyterHub ≥2
        servers = user_data.get("servers", {})
        if servers:
            for name, server_info in servers.items():
                if server_info.get("ready"):
                    url = f"{JUPYTERHUB_URL}/user/{username}/lab"
                    print(f"Server ready → {url}")
                    ctx["ti"].xcom_push(key="server_url", value=url)
                    return

        # Legacy JupyterHub (server is a non-empty path string when ready)
        elif isinstance(user_data.get("server"), str) and user_data["server"]:
            url = f"{JUPYTERHUB_URL}/user/{username}/lab"
            print(f"Server ready (legacy) → {url}")
            ctx["ti"].xcom_push(key="server_url", value=url)
            return

        print(f"[{attempt+1}/60] Waiting for server to be ready…")
        time.sleep(5)

    raise TimeoutError(f"Server for '{username}' did not become ready within 5 minutes.")


with DAG(
    dag_id="launch_notebook_dag",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["notebook"],
) as dag:

    username = "{{ dag_run.conf['username'] }}"
    profile  = "{{ dag_run.conf.get('profile', 'CPU — 4 cores, 16GB RAM') }}"

    t1 = PythonOperator(task_id="check_user_exists",  python_callable=check_user_exists,  op_kwargs={"username": username})
    t2 = PythonOperator(task_id="stop_stale_server",  python_callable=stop_stale_server,  op_kwargs={"username": username})
    t3 = PythonOperator(task_id="spawn_server",       python_callable=spawn_server,       op_kwargs={"username": username, "profile": profile})
    t4 = PythonOperator(task_id="poll_until_ready",   python_callable=poll_until_ready,   op_kwargs={"username": username})

    t1 >> t2 >> t3 >> t4