from datetime import timedelta
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.sensors.python import PythonSensor


def _wait_for_file(folder_name: str):
    return os.path.exists(f"/opt/airflow/data/raw/{folder_name}/data.csv")


default_args = {
    "owner": "airflow",
    "email": ["spb-312@yandex.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
        dag_id="2_dag_train_model",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:

    file_sensor = PythonSensor(
        task_id="file_sensor",
        python_callable=_wait_for_file,
        op_kwargs={"folder_name": "{{ ds }}"},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    preprocess = DockerOperator(
        task_id="docker-airflow-train-preprocess",
        image="airflow-train-preprocess",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/preprocessed/{{ ds }} --mode=train",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{Variable.get('data_folder_path')}:/data"]
    )

    split = DockerOperator(
        task_id="docker-airflow-train-split",
        image="airflow-train-split",
        command="--dir /data/preprocessed/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{Variable.get('data_folder_path')}:/data"]
    )

    train = DockerOperator(
        task_id="docker-airflow-train-model",
        image="airflow-train-model",
        command="--data_dir /data/preprocessed/{{ ds }} --model_dir /data/model/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{Variable.get('data_folder_path')}:/data"]
    )

    validate = DockerOperator(
        task_id="docker-airflow-train-validate",
        image="airflow-train-validate",
        command="--data_dir /data/preprocessed/{{ ds }} --model_dir /data/model/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{Variable.get('data_folder_path')}:/data"]
    )

    file_sensor >> preprocess >> split >> train >> validate


