Репозиторий содержит код для запуска web сервиса, решающего задачу классификации.

Запуск сервиса из репозитория:
<br> - установка зависимостей: pip install -e.
<br> - тест: pytest tests/test_app.py
<br> - запуск сервиса: cd online_inference; FEATURES_TRANSFORMER_PATH="features_transformer/features_transformer.pkl" MODEL_PATH="model/model.pkl" uvicorn app:app
<br> - отправка запросов: python make_request.py

Запуск сервиса из контейнера:
<br> - скачать: docker pull andrebelenko/made_prod_web_service_hw2
<br> - запустить: docker run --expose=8000 -p 8000:8000 andrebelenko/made_prod_web_service_hw2
<br> - протестировать: python make_request.py

Задание|Оценка
--- | --- 
1|3
2|3
3|2
4|3*
5|4
6|0
7|2
8|1
9|1
Общий балл 19

*Валидация типов данных, последовательности колонок: была ранее реализована в трансформере: см ml_project -> features -> build_features.py -> Features_transformer.check_input_df.
Валидация формирования датафрейма: см. endpoint "predict/" с использованием HTTPException