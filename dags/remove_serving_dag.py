from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import psycopg2
import logging
import os

logger = logging.getLogger(__name__)

DB_HOST = os.environ.get("MLFLOW_DB_HOST", "mlflow-postgresql-svc.mlops-mlflow.svc.cluster.local")
DB_PORT = os.environ.get("MLFLOW_DB_PORT", "5432")
DB_NAME = os.environ.get("MLFLOW_DB_NAME", "mlflow")
DB_USER = os.environ.get("MLFLOW_DB_USER", "mlflow")
DB_PASS = os.environ.get("MLFLOW_DB_PASSWORD", "mlflow-password")


def _get_job(dag_run_id: str) -> dict | None:
    with psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
        connect_timeout=5, options="-c lc_messages=C",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pod_name, svc_name, model_name, researcher_id
                FROM   researcher_serving_jobs
                WHERE  dag_run_id = %s
                """,
                (dag_run_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return {
        "pod_name":      row[0],
        "svc_name":      row[1],
        "model_name":    row[2],
        "researcher_id": row[3],
    }


def delete_k8s_resources(dag_run_id: str, **ctx):
    from kubernetes import client as k8s, config as k8s_config

    job = _get_job(dag_run_id)
    if job is None:
        logger.warning("No DB record found for dag_run_id=%s — nothing to delete", dag_run_id)
        return

    pod_name  = job["pod_name"]   # this is the Deployment name
    svc_name  = job["svc_name"]
    namespace = "mlops-serving"

    k8s_config.load_incluster_config()
    apps_v1        = k8s.AppsV1Api()
    core_v1        = k8s.CoreV1Api()
    autoscaling_v2 = k8s.AutoscalingV2Api()

    # ── HPA first ────────────────────────────────────────────────────────────
    if pod_name:
        hpa_name = f"{pod_name}-hpa"
        try:
            autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(hpa_name, namespace)
            logger.info("✅ Deleted HPA %s", hpa_name)
        except k8s.exceptions.ApiException as e:
            if e.status == 404:
                logger.info("HPA %s not found — skipping", hpa_name)
            else:
                raise

    # ── Deployment ────────────────────────────────────────────────────────────
    if pod_name:
        try:
            apps_v1.delete_namespaced_deployment(pod_name, namespace)
            logger.info("✅ Deleted Deployment %s", pod_name)
        except k8s.exceptions.ApiException as e:
            if e.status == 404:
                logger.info("Deployment %s not found — skipping", pod_name)
            else:
                raise

    # ── Service ───────────────────────────────────────────────────────────────
    if svc_name:
        try:
            core_v1.delete_namespaced_service(svc_name, namespace)
            logger.info("✅ Deleted Service %s", svc_name)
        except k8s.exceptions.ApiException as e:
            if e.status == 404:
                logger.info("Service %s not found — skipping", svc_name)
            else:
                raise

    logger.info("=== K8s resources removed for dag_run_id=%s ===", dag_run_id)


def delete_db_record(dag_run_id: str, **ctx):
    with psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
        connect_timeout=5, options="-c lc_messages=C",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM researcher_serving_jobs WHERE dag_run_id = %s",
                (dag_run_id,),
            )
        conn.commit()
    logger.info("✅ DB record deleted for dag_run_id=%s", dag_run_id)


with DAG(
    dag_id="remove_serving_dag",
    default_args={"owner": "mlops", "retries": 1},
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["serving"],
) as dag:

    dag_run_id_param = "{{ dag_run.conf['target_dag_run_id'] }}"

    t1 = PythonOperator(
        task_id="delete_k8s_resources",
        python_callable=delete_k8s_resources,
        op_kwargs={"dag_run_id": dag_run_id_param},
    )

    t2 = PythonOperator(
        task_id="delete_db_record",
        python_callable=delete_db_record,
        op_kwargs={"dag_run_id": dag_run_id_param},
    )

    t1 >> t2