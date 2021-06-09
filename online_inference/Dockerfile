FROM python:3.8

WORKDIR .

COPY ./online_inference ./online_inference
COPY ./models ./models
COPY ./features_transformer ./features_transformer
COPY ./ml_project/entities/feature_params.py ./ml_project/entities/feature_params.py
COPY ./ml_project/features ./ml_project/features
COPY ./ml_project/inference ./ml_project/inference
COPY ./ml_project/synthetic_data_generator.py ./ml_project/synthetic_data_generator.py

RUN pip install -r ./online_inference/requirements.txt

ENV FEATURES_TRANSFORMER_PATH="features_transformer/features_transformer.pkl"
ENV MODEL_PATH="models/model.pkl"
CMD ["uvicorn",  "online_inference.app:app", "--host", "0.0.0.0", "--port", "8000"]