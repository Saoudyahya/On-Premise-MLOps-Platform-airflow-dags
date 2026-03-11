from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import boto3
import subprocess
import os

default_args = {"owner": "mlops", "retries": 1}


def stage_raw_file(dataset, researcher_id, **ctx):
    """Upload the raw file to S3 under the researcher's own prefix."""
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3_key = f"{researcher_id}/{dataset}"          # e.g. "alice/train.csv"
    s3.upload_file(f"/data/incoming/{dataset}", "mlops-raw", s3_key)
    print(f"Staged s3://mlops-raw/{s3_key}")
    ctx["ti"].xcom_push(key="s3_key", value=s3_key)


def dvc_add(dataset, researcher_id, **ctx):
    """
    Copy the incoming file into the researcher's scoped folder inside the
    DVC workspace, then track it with `dvc add`.

    Resulting layout:
        dvc-workspace/
          data/
            <researcher_id>/
              <dataset>          ← tracked by DVC
              <dataset>.dvc      ← DVC pointer file
    """
    workspace = "/opt/airflow/dvc-workspace"
    dest_dir  = os.path.join(workspace, "data", researcher_id)
    os.makedirs(dest_dir, exist_ok=True)

    src  = f"/data/incoming/{dataset}"
    dest = os.path.join(dest_dir, dataset)

    # Copy only if the file isn't already in the workspace
    if not os.path.exists(dest):
        import shutil
        shutil.copy2(src, dest)
        print(f"Copied {src} → {dest}")

    dvc_path = os.path.join("data", researcher_id, dataset)   # relative to workspace
    subprocess.run(
        ["dvc", "add", dvc_path],
        check=True,
        cwd=workspace,
    )
    print(f"DVC tracking: {dvc_path}")
    ctx["ti"].xcom_push(key="dvc_path", value=dvc_path)


def git_commit(dataset, researcher_id, **ctx):
    """
    Commit the generated .dvc pointer file so the DVC lineage is captured
    in Git.  Uses a no-op if there is nothing to commit (idempotent).
    """
    workspace  = "/opt/airflow/dvc-workspace"
    dvc_path   = ctx["ti"].xcom_pull(key="dvc_path")
    dvc_file   = f"{dvc_path}.dvc"
    gitignore   = os.path.join("data", researcher_id, ".gitignore")

    subprocess.run(["git", "config", "user.email", "airflow@mlops"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name",  "Airflow"],       cwd=workspace, check=True)
    subprocess.run(["git", "add", dvc_file, gitignore],              cwd=workspace, check=True)

    result = subprocess.run(
        ["git", "commit", "-m",
         f"[ingest] {researcher_id}/{dataset} – tracked by DVC"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):   # 1 = nothing to commit → fine
        raise subprocess.CalledProcessError(result.returncode, "git commit", result.stderr)
    print(result.stdout or "Nothing new to commit.")


def dvc_push(**ctx):
    """Push all cached data to the DVC remote (MinIO / S3)."""
    subprocess.run(
        ["dvc", "push"],
        check=True,
        cwd="/opt/airflow/dvc-workspace",
    )
    print("DVC push complete.")


with DAG(
    dag_id="ingest_dag",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ingestion"],
    doc_md="""
## ingest_dag

Ingests a raw dataset file and tracks it with DVC, **scoped per researcher**.

### Trigger parameters (`dag_run.conf`)
| key             | required | example            | description                          |
|-----------------|----------|--------------------|--------------------------------------|
| `dataset`       | ✅       | `"train.csv"`      | Filename present in `/data/incoming/`|
| `researcher_id` | ✅       | `"alice"`          | Researcher username / unique ID      |

### What it does
1. **stage_raw_file** – uploads `/data/incoming/<dataset>` to  
   `s3://mlops-raw/<researcher_id>/<dataset>`
2. **dvc_add** – places the file under  
   `dvc-workspace/data/<researcher_id>/<dataset>` and runs `dvc add`
3. **git_commit** – commits the `.dvc` pointer file so lineage is in Git
4. **dvc_push** – pushes data to the DVC remote (MinIO)
""",
) as dag:

    dataset       = "{{ dag_run.conf['dataset'] }}"
    researcher_id = "{{ dag_run.conf['researcher_id'] }}"

    t1 = PythonOperator(
        task_id="stage_raw_file",
        python_callable=stage_raw_file,
        op_kwargs={"dataset": dataset, "researcher_id": researcher_id},
    )

    t2 = PythonOperator(
        task_id="dvc_add",
        python_callable=dvc_add,
        op_kwargs={"dataset": dataset, "researcher_id": researcher_id},
    )

    t3 = PythonOperator(
        task_id="git_commit",
        python_callable=git_commit,
        op_kwargs={"dataset": dataset, "researcher_id": researcher_id},
    )

    t4 = PythonOperator(
        task_id="dvc_push",
        python_callable=dvc_push,
    )

    t1 >> t2 >> t3 >> t4