# dags/mlflow_user_setup.py

import secrets
import requests
from datetime import datetime
from airflow.decorators import dag, task
from airflow.models import Variable

MLFLOW_URL   = "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000"
ADMIN_AUTH   = ("admin", "password")


@dag(
    dag_id="mlflow_user_setup",
    description="Create an MLflow user, experiment, and model registry for a researcher",
    schedule=None,          # manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={"username": "researcher_abc"},  # override at trigger time
    tags=["mlflow", "onboarding"],
)
def mlflow_user_setup():

    @task
    def create_user(username: str):
        password = secrets.token_urlsafe(16)

        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/users/create",
            json={"username": username, "password": password},
            auth=ADMIN_AUTH,
        )
        r.raise_for_status()
        print(f"✅ User created: {username}")

        # Save credentials to Airflow Variables
        Variable.set(f"mlflow_creds_{username}", f"{username}:{password}")
        print(f"✅ Credentials saved → mlflow_creds_{username}")

        return password

    @task
    def create_experiment(username: str, password: str):
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/create",
            json={"name": username},
            auth=ADMIN_AUTH,
        )
        r.raise_for_status()
        exp_id = r.json()["experiment_id"]
        print(f"✅ Experiment created: {username} (id={exp_id})")

        # Grant MANAGE permission
        requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/permissions/create",
            json={
                "experiment_id": exp_id,
                "username": username,
                "permission": "MANAGE",
            },
            auth=ADMIN_AUTH,
        ).raise_for_status()
        print(f"✅ Experiment permission granted")

        return exp_id

    @task
    def create_model_registry(username: str):
        model_name = f"{username}_models"

        # Create a placeholder registered model
        requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/create",
            json={"name": model_name},
            auth=ADMIN_AUTH,
        ).raise_for_status()
        print(f"✅ Model registry created: {model_name}")

        # Grant MANAGE permission
        requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/permissions/create",
            json={
                "name": model_name,
                "username": username,
                "permission": "MANAGE",
            },
            auth=ADMIN_AUTH,
        ).raise_for_status()
        print(f"✅ Model registry permission granted")

    @task
    def print_summary(username: str, exp_id: str):
        creds = Variable.get(f"mlflow_creds_{username}")
        _, password = creds.split(":")
        print("=" * 50)
        print(f"  MLflow user ready for: {username}")
        print(f"  Experiment:     {username}")
        print(f"  Model registry: {username}_models")
        print(f"  Username:       {username}")
        print(f"  Password:       {password}")
        print(f"  Stored in:      Airflow Variable → mlflow_creds_{username}")
        print("=" * 50)

    # ── wiring ────────────────────────────────────────────────
    username = "{{ params.username }}"

    password   = create_user(username)
    exp_id     = create_experiment(username, password)
    _registry  = create_model_registry(username)
    print_summary(username, exp_id)


mlflow_user_setup()