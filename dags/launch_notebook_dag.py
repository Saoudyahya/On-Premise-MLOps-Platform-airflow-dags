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


def _upsert_notebook_record(
    researcher_id: str,
    notebook_name: str,
    username: str,
    status: str,
    notebook_url: str = "",
    dag_run_id: str = "",
) -> None:
    with psycopg2.connect(HUB_DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO researcher_notebooks
                    (researcher_id, notebook_name, jupyter_username,
                     notebook_url, status, dag_run_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (researcher_id, notebook_name) DO UPDATE
                    SET status       = EXCLUDED.status,
                        notebook_url = EXCLUDED.notebook_url,
                        dag_run_id   = EXCLUDED.dag_run_id,
                        updated_at   = NOW()
                """,
                (researcher_id, notebook_name, username, notebook_url, status, dag_run_id),
            )
        conn.commit()
    logger.info(f"researcher_notebooks updated: {researcher_id}/{notebook_name} → {status}")


def spawn_named_server(researcher_id, notebook_name,
                       cpu_request, cpu_limit,
                       memory_request, memory_limit, **ctx):
    """
    Start a JupyterHub named server with the requested resource profile.

    Resources are passed as KubeSpawner user_options so JupyterHub's
    profile_list or kubespawner_override can apply them to the pod spec.
    Requires the following in jupyterhub/values.yaml:

        hub:
          extraConfig:
            resource-overrides: |
              from kubespawner import KubeSpawner

              def apply_resource_profile(spawner):
                  opts = spawner.user_options or {}
                  cpu_req = opts.get("cpu_request", "500m")
                  cpu_lim = opts.get("cpu_limit",   "1000m")
                  mem_req = opts.get("memory_request", "1Gi")
                  mem_lim = opts.get("memory_limit",   "1Gi")
                  spawner.cpu_guarantee  = _cpu_to_float(cpu_req)
                  spawner.cpu_limit      = _cpu_to_float(cpu_lim)
                  spawner.mem_guarantee  = mem_req
                  spawner.mem_limit      = mem_lim

              def _cpu_to_float(s):
                  if str(s).endswith("m"):
                      return float(s[:-1]) / 1000
                  return float(s)

              c.KubeSpawner.pre_spawn_hook = apply_resource_profile
    """
    logger.info(
        f"=== spawn_named_server | researcher={researcher_id} notebook={notebook_name} "
        f"cpu={cpu_request}-{cpu_limit} mem={memory_request}-{memory_limit} ==="
    )

    username   = _get_username(researcher_id)
    token      = get_admin_token()
    headers    = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    dag_run_id = ctx["dag_run"].run_id

    _upsert_notebook_record(
        researcher_id=researcher_id,
        notebook_name=notebook_name,
        username=username,
        status="starting",
        dag_run_id=dag_run_id,
    )

    r = requests.get(f"{JUPYTERHUB_API}/users/{username}", headers=headers, timeout=10)
    if r.status_code == 404:
        raise RuntimeError(
            f"User '{username}' not found in JupyterHub. "
            "Run POST /api/notebook/setup first to create the user."
        )
    r.raise_for_status()

    servers = r.json().get("servers", {})

    if notebook_name in servers:
        server_state = servers[notebook_name]
        if server_state.get("ready"):
            logger.info(f"Named server '{notebook_name}' already running — skipping spawn")
            _upsert_notebook_record(
                researcher_id=researcher_id,
                notebook_name=notebook_name,
                username=username,
                status="running",
                notebook_url=f"{JUPYTERHUB_PUBLIC_URL}/user/{username}/{notebook_name}/lab",
                dag_run_id=dag_run_id,
            )
            return
        pending = server_state.get("pending")
        logger.info(f"Named server '{notebook_name}' already pending={pending} — letting poll task wait")
        return

    # ── Spawn with resource user_options ──────────────────────────────────────
    # JupyterHub forwards user_options to KubeSpawner.  The pre_spawn_hook
    # (configured in extraConfig above) reads them and applies resource limits.
    spawn_body = {
    "cpu_request":    cpu_request,
    "cpu_limit":      cpu_limit,
    "memory_request": memory_request,
    "memory_limit":   memory_limit,
    }

    logger.info(
        f"Spawning: POST /hub/api/users/{username}/servers/{notebook_name} "
        f"body={spawn_body}"
    )
    r = requests.post(
        f"{JUPYTERHUB_API}/users/{username}/servers/{notebook_name}",
        headers=headers,
        json=spawn_body,
        timeout=30,
    )

    logger.info(f"Spawn response: {r.status_code} — {r.text}")

    if r.status_code in (200, 201, 202):
        logger.info(f"Named server '{notebook_name}' spawn accepted (HTTP {r.status_code})")
    else:
        raise RuntimeError(
            f"JupyterHub rejected spawn for '{username}/{notebook_name}': "
            f"HTTP {r.status_code} — {r.text}"
        )

    logger.info("=== spawn_named_server DONE ===")


def poll_until_ready(researcher_id, notebook_name, **ctx):
    """Poll until the named server is ready, then update the DB."""
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

            ctx["ti"].xcom_push(key="notebook_url",  value=notebook_url)
            ctx["ti"].xcom_push(key="notebook_name", value=notebook_name)

            logger.info(f"  notebook_url: {notebook_url}")
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

    researcher_id  = "{{ dag_run.conf['researcher_id'] }}"
    notebook_name  = "{{ dag_run.conf['notebook_name'] }}"
    cpu_request    = "{{ dag_run.conf.get('cpu_request',    '500m')  }}"
    cpu_limit      = "{{ dag_run.conf.get('cpu_limit',      '1000m') }}"
    memory_request = "{{ dag_run.conf.get('memory_request', '1Gi')   }}"
    memory_limit   = "{{ dag_run.conf.get('memory_limit',   '1Gi')   }}"

    t1 = PythonOperator(
        task_id="spawn_named_server",
        python_callable=spawn_named_server,
        op_kwargs={
            "researcher_id":  researcher_id,
            "notebook_name":  notebook_name,
            "cpu_request":    cpu_request,
            "cpu_limit":      cpu_limit,
            "memory_request": memory_request,
            "memory_limit":   memory_limit,
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