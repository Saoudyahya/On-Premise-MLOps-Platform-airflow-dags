from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import psycopg2
import time
import os
import logging

logger = logging.getLogger(__name__)

JUPYTERHUB_API = "http://hub.mlops-jupyterhub.svc.cluster.local:8081/hub/api"

JUPYTERHUB_PUBLIC_URL = os.environ.get(
    "JUPYTERHUB_PUBLIC_URL",
    "http://proxy-public.mlops-jupyterhub.svc.cluster.local:80",
)

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


def _get_username(researcher_id: str) -> str:
    """
    Look up the JupyterHub username for a researcher.
    Raises RuntimeError if jupyter_user_setup_dag hasn't been run yet.
    """
    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT jupyter_username FROM researcher_credentials WHERE researcher_id = %s",
                (researcher_id,),
            )
            row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"No JupyterHub credentials found for researcher '{researcher_id}'. "
            "Run POST /api/notebook/setup (jupyter_user_setup_dag) first."
        )
    return row[0]


def _upsert_notebook_record(
    researcher_id: str,
    notebook_name: str,
    username: str,
    status: str,
    notebook_url: str = "",
    dag_run_id: str = "",
) -> None:
    """
    Insert or update a row in researcher_notebooks so Django can list
    notebooks without calling the JupyterHub API.
    """
    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO researcher_notebooks
                    (researcher_id, notebook_name, jupyter_username,
                     notebook_url, status, dag_run_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (researcher_id, notebook_name) DO UPDATE
                    SET status           = EXCLUDED.status,
                        notebook_url     = EXCLUDED.notebook_url,
                        dag_run_id       = EXCLUDED.dag_run_id,
                        updated_at       = NOW()
                """,
                (researcher_id, notebook_name, username, notebook_url, status, dag_run_id),
            )
        conn.commit()
    logger.info(f"researcher_notebooks updated: {researcher_id}/{notebook_name} → {status}")


def spawn_named_server(researcher_id, notebook_name, **ctx):
    """
    Start a JupyterHub named server for this researcher.
    The user must already exist (jupyter_user_setup_dag must have run).
    """
    logger.info(f"=== spawn_named_server | researcher={researcher_id} notebook={notebook_name} ===")

    username    = _get_username(researcher_id)
    token       = get_admin_token()
    headers     = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    dag_run_id  = ctx["dag_run"].run_id

    # Record that we're starting this notebook
    _upsert_notebook_record(
        researcher_id=researcher_id,
        notebook_name=notebook_name,
        username=username,
        status="starting",
        dag_run_id=dag_run_id,
    )

    # Check if this named server already exists and is ready
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
    r.raise_for_status()

    servers = r.json().get("servers", {})
    if notebook_name in servers and servers[notebook_name].get("ready"):
        logger.info(f"Named server '{notebook_name}' already running — skipping spawn")
        return

    # Start the named server
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/servers/{notebook_name}",
        headers=headers,
        json={},
        timeout=10,
    )
    logger.info(f"Spawn response: {r.status_code}")

    # 201 = created, 202 = pending, 400 = already running
    if r.status_code not in (201, 202, 400):
        r.raise_for_status()

    logger.info("=== spawn_named_server DONE ===")


def poll_until_ready(researcher_id, notebook_name, **ctx):
    """
    Poll the Hub API until the named server is ready, then update the
    researcher_notebooks table so Django can return the URL immediately.
    """
    logger.info(f"=== poll_until_ready | researcher={researcher_id} notebook={notebook_name} ===")

    username = _get_username(researcher_id)
    token    = get_admin_token()
    headers  = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    for attempt in range(60):
        logger.info(f"[{attempt + 1}/60] Polling named server '{notebook_name}'...")

        r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
        r.raise_for_status()

        server = r.json().get("servers", {}).get(notebook_name)

        if server is None:
            logger.info("  Server not found yet — waiting...")
            time.sleep(5)
            continue

        ready   = server.get("ready", False)
        pending = server.get("pending")
        logger.info(f"  ready={ready}, pending={pending}")

        if ready:
            notebook_url = f"{JUPYTERHUB_PUBLIC_URL}/user/{username}/{notebook_name}/lab"

            _upsert_notebook_record(
                researcher_id=researcher_id,
                notebook_name=notebook_name,
                username=username,
                status="running",
                notebook_url=notebook_url,
                dag_run_id=ctx["dag_run"].run_id,
            )

            logger.info("=" * 60)
            logger.info(f"  researcher_id : {researcher_id}")
            logger.info(f"  notebook_name : {notebook_name}")
            logger.info(f"  notebook_url  : {notebook_url}")
            logger.info("=" * 60)
            logger.info("=== poll_until_ready DONE ===")
            return

        time.sleep(5)

    raise TimeoutError(
        f"Named server '{notebook_name}' for '{username}' did not become ready within 5 minutes."
    )


with DAG(
    dag_id="launch_notebook_dag",
    default_args={"owner": "mlops", "retries": 1},
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["notebook"],
) as dag:

    researcher_id = "{{ dag_run.conf['researcher_id'] }}"
    notebook_name = "{{ dag_run.conf['notebook_name'] }}"

    t1 = PythonOperator(
        task_id="spawn_named_server",
        python_callable=spawn_named_server,
        op_kwargs={
            "researcher_id": researcher_id,
            "notebook_name": notebook_name,
        },
    )

    t2 = PythonOperator(
        task_id="poll_until_ready",
        python_callable=poll_until_ready,
        op_kwargs={
            "researcher_id": researcher_id,
            "notebook_name": notebook_name,
        },
    )

    t1 >> t2