FROM python:3.8

COPY ./requirements.txt /
COPY ./configs ./configs
COPY ./data ./data
COPY ./features_transformer ./features_transformer
COPY ./ml_project ./ml_project
COPY ./models ./models

RUN pip install -r requirements.txt

WORKDIR .

CMD ["python", "python ml_project/inference_pipeline.py configs/config.yaml"]