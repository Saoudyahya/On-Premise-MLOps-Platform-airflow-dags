from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import time
import os

default_args = {"owner": "mlops", "retries": 1}

JUPYTERHUB_URL = "http://proxy-public.mlops-jupyterhub.svc.cluster.local:80"
JUPYTERHUB_API = f"{JUPYTERHUB_URL}/hub/api"
ADMIN_TOKEN = os.environ.get("JUPYTERHUB_ADMIN_TOKEN", "")

HEADERS = {
    "Authorization": f"token {ADMIN_TOKEN}",
    "Content-Type": "application/json",
}

def check_user_exists(username, **ctx):
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS)
    if r.status_code == 404:
        requests.post(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS).raise_for_status()
    elif r.status_code != 200:
        r.raise_for_status()

def stop_stale_server(username, **ctx):
    r = requests.delete(f"{JUPYTERHUB_API}/users/{username}/server", headers=HEADERS)
    if r.status_code not in (204, 404):
        r.raise_for_status()

def spawn_server(username, profile, **ctx):
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/server",
        headers=HEADERS,
        json={"profile_name": profile},
    )
    if r.status_code not in (201, 202):
        r.raise_for_status()

def poll_until_ready(username, **ctx):
    for _ in range(60):
        r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=HEADERS)
        r.raise_for_status()
        server = r.json().get("server")
        if server and r.json().get("server", {}) != {}:
            print(f"Server ready: {JUPYTERHUB_URL}/user/{username}/lab")
            return
        time.sleep(5)
    raise TimeoutError(f"Server for {username} did not become ready in time")

with DAG(
    dag_id="launch_notebook_dag",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["notebook"],
) as dag:

    username = "{{ dag_run.conf['username'] }}"
    profile  = "{{ dag_run.conf.get('profile', 'CPU — 4 cores, 16GB RAM') }}"

    t1 = PythonOperator(
        task_id="check_user_exists",
        python_callable=check_user_exists,
        op_kwargs={"username": username},
    )

    t2 = PythonOperator(
        task_id="stop_stale_server",
        python_callable=stop_stale_server,
        op_kwargs={"username": username},
    )

    t3 = PythonOperator(
        task_id="spawn_server",
        python_callable=spawn_server,
        op_kwargs={"username": username, "profile": profile},
    )

    t4 = PythonOperator(
        task_id="poll_until_ready",
        python_callable=poll_until_ready,
        op_kwargs={"username": username},
    )

    t1 >> t2 >> t3 >> t4