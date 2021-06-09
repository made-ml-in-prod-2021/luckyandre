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

