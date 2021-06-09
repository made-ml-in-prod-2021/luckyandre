Я еще не завершил задание.
Отправьте, пожалуйста, мое решение на доработку.

<br> export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
<br> docker compose up --build --remove-orphans
