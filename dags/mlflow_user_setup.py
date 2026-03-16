# dags/mlflow_user_setup.py
# SIMPLIFIED: only provisions the MLflow user account.
# Experiment + model registry are created by the researcher from their notebook.

import secrets
import psycopg2
import requests
from datetime import datetime
from airflow.sdk import dag, task

MLFLOW_URL = "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000"
ADMIN_USER = "admin"
ADMIN_PASS = "mlops-admin-2024"   # must match auth.ini → admin_password

DB_DSN = (
    "host=mlflow-postgresql-svc.mlops-mlflow.svc.cluster.local "
    "port=5432 "
    "dbname=mlflow "
    "user=mlflow "
    "password=mlflow-password "
    "sslmode=disable"
)


def _upsert_credentials(researcher_id: str, username: str, password: str) -> None:
    with psycopg2.connect(DB_DSN) as conn:
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
    description="Provision an MLflow user account for a researcher",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={"username": "researcher_abc"},
    tags=["mlflow", "onboarding"],
)
def mlflow_user_setup():

    @task
    def create_mlflow_user(**context):
        username = context["params"]["username"]
        password = secrets.token_urlsafe(16)

        # ── 1. Verify admin credentials are actually working ───────────────────
        # Probe the admin user endpoint first — cheap read that confirms auth.
        # If this 403s, the ADMIN_PASS in this file doesn't match auth.ini.
        probe = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/users/get",
            params={"username": ADMIN_USER},
            auth=(ADMIN_USER, ADMIN_PASS),
            timeout=10,
        )
        if probe.status_code == 403:
            raise RuntimeError(
                "Admin credentials rejected by MLflow (403). "
                "Verify ADMIN_PASS matches auth.ini → admin_password on the MLflow pod.\n"
                "Quick check:\n"
                "  kubectl exec -n mlops-mlflow deploy/mlflow -- "
                "  curl -su admin:mlops-admin-2024 "
                "  http://localhost:5000/api/2.0/mlflow/users/get?username=admin"
            )
        print(f"✅ Admin auth probe: HTTP {probe.status_code}")

        # ── 2. Create researcher user (or rotate password if they exist) ───────
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/users/create",
            json={"username": username, "password": password},
            auth=(ADMIN_USER, ADMIN_PASS),
            timeout=10,
        )

        if r.status_code == 200:
            print(f"✅ MLflow user created: {username}")

        elif r.status_code == 409:
            # User already exists — rotate the password so re-running the DAG
            # always produces fresh credentials rather than silently failing.
            print(f"⚠️  User '{username}' already exists — rotating password")
            r2 = requests.patch(
                f"{MLFLOW_URL}/api/2.0/mlflow/users/update-password",
                json={"username": username, "password": password},
                auth=(ADMIN_USER, ADMIN_PASS),
                timeout=10,
            )
            r2.raise_for_status()
            print(f"✅ Password rotated for: {username}")

        else:
            # Surface the MLflow error body to make debugging easier
            raise RuntimeError(
                f"MLflow /users/create returned {r.status_code}: {r.text}"
            )

        # ── 3. Persist in PostgreSQL ───────────────────────────────────────────
        _upsert_credentials(
            researcher_id=username,
            username=username,
            password=password,
        )
        print("✅ Credentials stored in researcher_credentials (mlflow PostgreSQL)")

        print("=" * 60)
        print(f"  Researcher : {username}")
        print(f"  Password   : {password}")
        print(f"  Fetch creds: GET /api/mlflow/credentials/{username}")
        print()
        print("  The researcher creates their own experiment and model")
        print("  registry from the notebook — no admin action needed.")
        print("=" * 60)

        return {"username": username, "password": password}

    create_mlflow_user()


mlflow_user_setup()