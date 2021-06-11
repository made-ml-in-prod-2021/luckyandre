import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.sensors.python import PythonSensor


def _wait_for_file(pre_folder_name: str, folder_name: str, file_name: str):
    return os.path.exists(f"/opt/airflow/data/{pre_folder_name}/{folder_name}/{file_name}")


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
        dag_id="3_dag_inference",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:

    data_sensor = PythonSensor(
        task_id="data_sensor",
        python_callable=_wait_for_file,
        op_kwargs={
            "pre_folder_name": "raw",
            "folder_name": "{{ ds }}",
            "file_name": "data.csv"
        },
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    preprocess = DockerOperator(
        task_id="docker-airflow-inference-preprocess",
        image="airflow-train-preprocess",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/preprocessed/{{ ds }} --mode=inference",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{Variable.get('data_folder_path')}:/data"]
    )

    model_sensor = PythonSensor(
        task_id="model_sensor",
        python_callable=_wait_for_file,
        op_kwargs={
            "pre_folder_name": "model",
            "folder_name": "{{ var.value.model_folder_name }}",
            "file_name": "model.pkl"
        },
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    predict = DockerOperator(
        task_id="docker-airflow-inference-predict",
        image="airflow-inference-predict",
        command="--input_data_dir /data/preprocessed/{{ ds }} --output_data_dir /data/predicted/{{ ds }} --model_dir /data/model/{{ var.value.model_folder_name }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{Variable.get('data_folder_path')}:/data"]
    )

    data_sensor >> preprocess >> model_sensor >> predict
