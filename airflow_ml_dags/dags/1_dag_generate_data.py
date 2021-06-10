from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
        dag_id="1_dag_generate_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:

    start = DummyOperator(task_id="start")
    generate = DockerOperator(
        task_id="docker-airflow-generate-data",
        image="airflow-generate-data",
        command="--size=1000 --random_state={{ ds_nodash }} --output_dir=/data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=["/Users/a18648975/Desktop/HW3/airflow_ml_dags/data/:/data"] # TODO check this path
    )

    start >> generate