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


def stop_stale_server(username, **ctx):
    """Stop the server only if it is stuck in a pending/unknown state."""
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS)

    # User doesn't exist yet — nothing to stop
    if r.status_code == 404:
        print(f"User '{username}' does not exist yet, skipping stop.")
        return
    r.raise_for_status()

    user_data = r.json()
    servers   = user_data.get("servers", {})

    for name, server_info in servers.items():
        if server_info.get("ready"):
            print(f"Server '{name}' is healthy — leaving it alone.")
        else:
            print(f"Server '{name}' is stuck (pending={server_info.get('pending')}) — stopping it.")
            requests.delete(
                f"{JUPYTERHUB_API}/users/{username}/servers/{name}",
                headers=HEADERS,
            )
            time.sleep(3)


def spawn_server(username, **ctx):
    """
    Spawn a default server for the user.
    JupyterHub creates the user automatically on first spawn if they don't exist.
    Skips if a healthy server is already running.
    """
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS)

    if r.status_code == 200:
        servers = r.json().get("servers", {})
        for name, server_info in servers.items():
            if server_info.get("ready"):
                print(f"Server already running for '{username}' — skipping spawn.")
                return

    # POST to /server — JupyterHub auto-creates the user if needed
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/server",
        headers=HEADERS,
        json={},
    )
    if r.status_code not in (201, 202):
        r.raise_for_status()
    print(f"Spawn requested for '{username}'.")


def poll_until_ready(username, **ctx):
    """Poll every 5 s for up to 5 min until the server is ready."""
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

        # Legacy single-server API (server is a non-empty path string when ready)
        elif isinstance(user_data.get("server"), str) and user_data["server"]:
            url = f"{JUPYTERHUB_URL}/user/{username}/lab"
            print(f"Server ready (legacy API) → {url}")
            ctx["ti"].xcom_push(key="server_url", value=url)
            return

        print(f"[{attempt + 1}/60] Waiting for server to be ready…")
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

    t1 = PythonOperator(task_id="stop_stale_server", python_callable=stop_stale_server, op_kwargs={"username": username})
    t2 = PythonOperator(task_id="spawn_server",      python_callable=spawn_server,      op_kwargs={"username": username})
    t3 = PythonOperator(task_id="poll_until_ready",  python_callable=poll_until_ready,  op_kwargs={"username": username})

    t1 >> t2 >> t3