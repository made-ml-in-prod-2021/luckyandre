FROM python:3.8

WORKDIR .

COPY ./online_inference ./online_inference

RUN pip install -r ./online_inference/requirements.txt

ENV FEATURES_TRANSFORMER_PATH="online_inference/features_transformer/features_transformer.pkl"
ENV MODEL_PATH="online_inference/model/model.pkl"
CMD ["uvicorn",  "online_inference.app:app", "--host", "0.0.0.0", "--port", "8000"]