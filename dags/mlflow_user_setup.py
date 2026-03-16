# dags/mlflow_user_setup.py

import secrets
import psycopg2
import requests
from datetime import datetime
from airflow.sdk import dag, task

MLFLOW_URL = "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000"
ADMIN_AUTH = ("admin", "mlops-admin-2024")

# Same PostgreSQL that MLflow uses — researcher credentials live in a
# separate table so Airflow Variables are never involved.
DB_DSN = (
    "host=mlflow-postgresql-svc.mlops-mlflow.svc.cluster.local "
    "port=5432 "
    "dbname=mlflow "
    "user=mlflow "
    "password=mlflow-password "
    "sslmode=disable"
)


def _get_db():
    return psycopg2.connect(DB_DSN)


def _upsert_credentials(researcher_id: str, username: str, password: str) -> None:
    """
    Insert or update researcher credentials in the researcher_credentials table.
    On conflict (same researcher_id) update password + updated_at so re-running
    the DAG rotates the password instead of crashing.
    """
    with _get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO researcher_credentials
                    (researcher_id, mlflow_username, mlflow_password)
                VALUES (%s, %s, %s)
                ON CONFLICT (researcher_id) DO UPDATE
                    SET mlflow_password = EXCLUDED.mlflow_password,
                        updated_at      = NOW()
                """,
                (researcher_id, username, password),
            )
        conn.commit()


@dag(
    dag_id="mlflow_user_setup",
    description="Create MLflow user, experiment and model registry for a researcher",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={"username": "researcher_abc"},
    tags=["mlflow", "onboarding"],
)
def mlflow_user_setup():

    @task
    def create_user(**context):
        username = context["params"]["username"]
        password = secrets.token_urlsafe(16)

        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/users/create",
            json={"username": username, "password": password},
            auth=ADMIN_AUTH,
        )
        r.raise_for_status()
        print(f"✅ MLflow user created: {username}")

        # ── Persist in PostgreSQL (not Airflow Variables) ──────────────────────
        _upsert_credentials(
            researcher_id=username,
            username=username,
            password=password,
        )
        print(f"✅ Credentials stored in PostgreSQL → researcher_credentials")

        return {"username": username, "password": password}

    @task
    def create_experiment(user_info: dict):
        username = user_info["username"]

        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/create",
            json={"name": username},
            auth=ADMIN_AUTH,
        )
        r.raise_for_status()
        exp_id = r.json()["experiment_id"]
        print(f"✅ Experiment created: {username} (id={exp_id})")

        requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/permissions/create",
            json={
                "experiment_id": exp_id,
                "username":      username,
                "permission":    "MANAGE",
            },
            auth=ADMIN_AUTH,
        ).raise_for_status()
        print("✅ Experiment MANAGE permission granted")

        return {**user_info, "exp_id": exp_id}

    @task
    def create_model_registry(user_info: dict):
        username   = user_info["username"]
        model_name = f"{username}_models"

        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/create",
            json={"name": model_name},
            auth=ADMIN_AUTH,
        )
        if r.status_code == 400:
            print(f"⚠️  Model registry '{model_name}' already exists — skipping")
        else:
            r.raise_for_status()
            print(f"✅ Model registry created: {model_name}")

        requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/permissions/create",
            json={
                "name":       model_name,
                "username":   username,
                "permission": "MANAGE",
            },
            auth=ADMIN_AUTH,
        ).raise_for_status()
        print("✅ Model registry MANAGE permission granted")

        return user_info

    @task
    def print_summary(user_info: dict):
        username = user_info["username"]

        # Re-read from DB so the summary reflects what was actually stored
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT mlflow_password FROM researcher_credentials "
                    "WHERE researcher_id = %s",
                    (username,),
                )
                row = cur.fetchone()

        password = row[0] if row else "(not found)"

        print("=" * 60)
        print(f"  MLflow user ready : {username}")
        print(f"  Experiment        : {username}")
        print(f"  Model registry    : {username}_models")
        print(f"  Username          : {username}")
        print(f"  Password          : {password}")
        print(f"  Credentials in    : researcher_credentials table")
        print(f"  Retrieve via      : GET /api/mlflow/credentials/{username}")
        print("=" * 60)

    # ── wiring ─────────────────────────────────────────────────────────────────
    user_info   = create_user()
    user_info_2 = create_experiment(user_info)
    user_info_3 = create_model_registry(user_info_2)
    print_summary(user_info_3)


mlflow_user_setup()