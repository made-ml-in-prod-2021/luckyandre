from datetime import timedelta
import random

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "1_dag_generate_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(2),
) as dag:
    start = DummyOperator(task_id="start")
    generate = DummyOperator(task_id="generate")
    # generate = DockerOperator(
    #     image="airflow-generate-data",
    #     command="--output_dir=/data/raw/{{ ds }} --size=1000 --random_state=7",
    #     network_mode="bridge",
    #     task_id="docker-airflow-generate-data",
    #     do_xcom_push=False,
    #     volumes=["/Users/a18648975/Desktop/HW3/:/data"]
    # )
    start >> generate