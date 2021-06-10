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
        dag_id="2_dag_train_model",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:

    preprocess = DockerOperator(
        task_id="docker-airflow-train-preprocess",
        image="airflow-train-preprocess",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/preprocessed/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=["/Users/a18648975/Desktop/HW3/airflow_ml_dags/data/:/data"] # TODO check this path
    )

    split = DockerOperator(
        task_id="docker-airflow-train-split",
        image="airflow-train-split",
        command="--dir /data/preprocessed/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=["/Users/a18648975/Desktop/HW3/airflow_ml_dags/data/:/data"] # TODO check this path
    )


    train = DummyOperator(task_id="train")
    validation = DummyOperator(task_id="validation")


    preprocess >> split >> train >> validation