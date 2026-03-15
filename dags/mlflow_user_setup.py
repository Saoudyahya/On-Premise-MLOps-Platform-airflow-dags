# dags/mlflow_user_setup.py

import secrets
import requests
from datetime import datetime
from airflow.sdk import dag, task
from airflow.models import Variable

MLFLOW_URL = "http://mlflow-svc.mlops-mlflow.svc.cluster.local:5000"
ADMIN_AUTH = ("admin", "password")


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
        print(f"✅ User created: {username}")

        Variable.set(f"mlflow_creds_{username}", f"{username}:{password}")
        print(f"✅ Credentials saved → mlflow_creds_{username}")

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
                "username": username,
                "permission": "MANAGE",
            },
            auth=ADMIN_AUTH,
        ).raise_for_status()
        print(f"✅ Experiment permission granted")

        return {**user_info, "exp_id": exp_id}

    @task
    def create_model_registry(user_info: dict):
        username  = user_info["username"]
        model_name = f"{username}_models"

        # Create the registered model
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/create",
            json={"name": model_name},
            auth=ADMIN_AUTH,
        )
        # 400 here usually means model already exists — that's fine
        if r.status_code == 400:
            print(f"⚠️  Model registry '{model_name}' already exists, skipping creation")
        else:
            r.raise_for_status()
            print(f"✅ Model registry created: {model_name}")

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

        return user_info

    @task
    def print_summary(user_info: dict):
        username = user_info["username"]
        creds    = Variable.get(f"mlflow_creds_{username}")
        _, password = creds.split(":")

        print("=" * 50)
        print(f"  MLflow user ready: {username}")
        print(f"  Experiment:        {username}")
        print(f"  Model registry:    {username}_models")
        print(f"  Username:          {username}")
        print(f"  Password:          {password}")
        print(f"  Stored in:         mlflow_creds_{username}")
        print("=" * 50)

    # ── wiring ────────────────────────────────────────────────
    user_info    = create_user()
    user_info_2  = create_experiment(user_info)
    user_info_3  = create_model_registry(user_info_2)
    print_summary(user_info_3)


mlflow_user_setup()