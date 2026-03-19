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
            f"No JupyterHub credentials found for researcher '{researcher_id}'."
        )
    return row[0]


def _mark_notebook_stopped(researcher_id: str, notebook_name: str) -> None:
    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE researcher_notebooks
                   SET status      = 'stopped',
                       notebook_url = '',
                       updated_at  = NOW()
                 WHERE researcher_id = %s AND notebook_name = %s
                """,
                (researcher_id, notebook_name),
            )
        conn.commit()
    logger.info(f"researcher_notebooks marked stopped: {researcher_id}/{notebook_name}")


def stop_named_server(researcher_id, notebook_name, **ctx):
    """
    Stop a JupyterHub named server and mark it stopped in the DB.

    Result is always pushed to XCom so Django never needs to query the DB:
      - result = "stopped"         → server was running, now stopped
      - result = "not_found"       → notebook doesn't exist in DB
      - result = "already_stopped" → notebook was already stopped in DB

    The DAG always succeeds for these known states; Django maps them to
    the right HTTP status codes.
    Unexpected Hub API or DB errors still raise and fail the DAG (→ 502).
    """
    logger.info(f"=== stop_named_server | researcher={researcher_id} notebook={notebook_name} ===")

    ti = ctx["ti"]

    username = _get_username(researcher_id)
    token    = get_admin_token()
    headers  = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    # ── 1. Check notebook exists in DB ────────────────────────────────────────
    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM researcher_notebooks WHERE researcher_id = %s AND notebook_name = %s",
                (researcher_id, notebook_name),
            )
            row = cur.fetchone()

    if row is None:
        logger.info(f"Notebook '{notebook_name}' not found in DB — nothing to remove")
        ti.xcom_push(key="result",  value="not_found")
        ti.xcom_push(key="message", value=f"Notebook '{notebook_name}' not found for researcher '{researcher_id}'")
        return

    if row[0] == "stopped":
        logger.info(f"Notebook '{notebook_name}' is already stopped — skipping")
        ti.xcom_push(key="result",  value="already_stopped")
        ti.xcom_push(key="message", value=f"Notebook '{notebook_name}' is already stopped")
        return

    # ── 2. Verify user exists in Hub ─────────────────────────────────────────
    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
    if r.status_code == 404:
        # User gone from Hub but DB row exists — clean up and report stopped
        logger.warning(f"User '{username}' not found in Hub — marking stopped anyway")
        _mark_notebook_stopped(researcher_id, notebook_name)
        ti.xcom_push(key="result",  value="stopped")
        ti.xcom_push(key="message", value=f"Notebook '{notebook_name}' cleaned up (user not in Hub)")
        return
    r.raise_for_status()

    servers = r.json().get("servers", {})

    # ── 3. If not running in Hub, just clean the DB row ──────────────────────
    if notebook_name not in servers:
        logger.info(f"Named server '{notebook_name}' not running in Hub — cleaning DB record")
        _mark_notebook_stopped(researcher_id, notebook_name)
        ti.xcom_push(key="result",  value="stopped")
        ti.xcom_push(key="message", value=f"Notebook '{notebook_name}' was not running; record cleaned up")
        return

    # ── 4. Send DELETE to stop the named server ───────────────────────────────
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

    # ── 5. Poll until the server disappears (max 2 min) ───────────────────────
    logger.info("Waiting for named server to stop (max 2 min)...")
    for attempt in range(24):
        time.sleep(5)
        r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
        r.raise_for_status()
        if notebook_name not in r.json().get("servers", {}):
            logger.info(f"✅ Named server '{notebook_name}' has stopped")
            break
        pending = r.json()["servers"][notebook_name].get("pending")
        logger.info(f"  [{attempt + 1}/24] Still stopping, pending={pending} — waiting...")
    else:
        raise TimeoutError(
            f"Named server '{notebook_name}' for '{username}' did not stop within 2 minutes."
        )

    # ── 6. Mark stopped in DB ─────────────────────────────────────────────────
    _mark_notebook_stopped(researcher_id, notebook_name)

    ti.xcom_push(key="result",  value="stopped")
    ti.xcom_push(key="message", value=f"Notebook '{notebook_name}' stopped successfully")

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