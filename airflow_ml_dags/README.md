Не успел добавить тесты (они практически готовы), поэтому отправьте, пожалуйста, мое решение на доработку и я добавлю финальную часть.

export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")

docker compose -f airflow_ml_dags/docker-compose.yml up --build --remove-orphans
