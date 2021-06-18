<br> Скрины работы дагов см в папке airflow_ml_dags/confirmation
<br>
<br> Запуск airflow:
<br> export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
<br> docker compose -f airflow_ml_dags/docker-compose.yml up --build --remove-orphans
<br>
<br> Необходимо установить переменные airflow (Admin -> Variables), например:
<br> data_folder_path = '/Users/a18648975/Desktop/HW3/airflow_ml_dags/data'
<br> model_folder_name = '2021-06-06'
<br>
Задание|Оценка
--- | --- 
1|5
2|10
3|5
3a|3
4|10
5|0
6|0
7|0
8|3
9|1
Общий балл|37
