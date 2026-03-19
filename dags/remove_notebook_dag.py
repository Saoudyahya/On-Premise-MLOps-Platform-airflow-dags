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
            "Run POST /api/notebook/setup first."
        )
    return row[0]


def _mark_notebook_stopped(researcher_id: str, notebook_name: str) -> None:
    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE researcher_notebooks
                   SET status     = 'stopped',
                       notebook_url = '',
                       updated_at = NOW()
                 WHERE researcher_id = %s AND notebook_name = %s
                """,
                (researcher_id, notebook_name),
            )
        conn.commit()
    logger.info(f"researcher_notebooks marked stopped: {researcher_id}/{notebook_name}")


def stop_named_server(researcher_id, notebook_name, **ctx):
    """
    Stop a running JupyterHub named server for this researcher.

    Steps:
      1. Verify the user exists in the Hub.
      2. If the named server does not exist or is already stopped, skip gracefully.
      3. Send DELETE to stop the server.
      4. Poll until the server disappears from the Hub API (max 2 min).
      5. Mark the notebook as 'stopped' in researcher_notebooks.
    """
    logger.info(f"=== stop_named_server | researcher={researcher_id} notebook={notebook_name} ===")

    username = _get_username(researcher_id)
    token    = get_admin_token()
    headers  = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    # ── 1. Check the user exists ──────────────────────────────────────────────
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)

    if r.status_code == 404:
        raise RuntimeError(
            f"User '{username}' not found in JupyterHub. "
            "Nothing to remove."
        )
    r.raise_for_status()

    servers = r.json().get("servers", {})

    # ── 2. Already gone — just clean up the DB record and exit ───────────────
    if notebook_name not in servers:
        logger.info(f"Named server '{notebook_name}' is not running — marking stopped in DB")
        _mark_notebook_stopped(researcher_id, notebook_name)
        logger.info("=== stop_named_server DONE (already stopped) ===")
        return

    # ── 3. Send DELETE to stop the named server ───────────────────────────────
    logger.info(f"Stopping: DELETE /hub/api/users/{username}/servers/{notebook_name}")
    r = requests.delete(
        f"{JUPYTERHUB_API}/users/{username}/servers/{notebook_name}",
        headers=headers,
        timeout=30,
    )

    logger.info(f"Stop response: {r.status_code} — {r.text}")

    # 204 = stopped, 202 = stopping in progress, 400 = already stopped
    if r.status_code not in (200, 202, 204, 400):
        raise RuntimeError(
            f"JupyterHub rejected stop for '{username}/{notebook_name}': "
            f"HTTP {r.status_code} — {r.text}"
        )

    # ── 4. Poll until the server disappears (max 2 min) ───────────────────────
    logger.info("Waiting for named server to stop (max 2 min)...")
    for attempt in range(24):
        time.sleep(5)
        r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
        r.raise_for_status()

        servers = r.json().get("servers", {})
        if notebook_name not in servers:
            logger.info(f"✅ Named server '{notebook_name}' has stopped")
            break

        pending = servers[notebook_name].get("pending")
        logger.info(f"  [{attempt + 1}/24] Still stopping, pending={pending} — waiting...")
    else:
        raise TimeoutError(
            f"Named server '{notebook_name}' for '{username}' did not stop within 2 minutes."
        )

    # ── 5. Mark stopped in DB ─────────────────────────────────────────────────
    _mark_notebook_stopped(researcher_id, notebook_name)

    logger.info("=" * 60)
    logger.info(f"  researcher_id : {researcher_id}")
    logger.info(f"  notebook_name : {notebook_name}")
    logger.info(f"  status        : stopped")
    logger.info("=" * 60)
    logger.info("=== stop_named_server DONE ===")


with DAG(
    dag_id="remove_notebook_dag",
    default_args={"owner": "mlops", "retries": 1},
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["notebook"],
) as dag:

    PythonOperator(
        task_id="stop_named_server",
        python_callable=stop_named_server,
        op_kwargs={
            "researcher_id": "{{ dag_run.conf['researcher_id'] }}",
            "notebook_name": "{{ dag_run.conf['notebook_name'] }}",
        },
    )
