Homework1 (pipeline):
<br />- установка: pip install -e .
<br />- тестирование: pytest tests
<br />- отчет: python reports/report.py configs/config.yaml
<br />- обучение: python ml_project/train_pipeline.py configs/config.yaml
<br />- предсказание: python ml_project/inference_pipeline.py configs/config.yaml

Homework2 (Docker, web service):
<br> - скачать: docker pull andrebelenko/made_prod_web_service_hw2
<br> - запустить: docker run --expose=8000 -p 8000:8000 andrebelenko/made_prod_web_service_hw2
<br> - отправка запросов: python online_inference/make_request.py

Homework3 (Airflow):
<br> запуск:
<br> export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
<br> docker compose -f airflow_ml_dags/docker-compose.yml up --build --remove-orphans
<br> необходимо установить переменные airflow (Admin -> Variables), например:
<br> data_folder_path = '/Users/a18648975/Desktop/HW3/airflow_ml_dags/data'
<br> model_folder_name = '2021-06-06'

