import logging
import os

import uvicorn
from fastapi import FastAPI

from ml_project import inference_pipeline_command
from ml_project.entities import read_params
from ml_project.inference import load_transformer, load_model

CONFIG_PATH = "configs/config.yaml"

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.get("/healz")
def health() -> bool:
    params = read_params(CONFIG_PATH)
    transformer = load_transformer(params.features_transformer_path)
    model = load_model(params.model_path)
    # TODO прочие проверки (включая данные)
    return (not (transformer is None)) and (not (model is None))


@app.get("/predict")
def predict():
    return inference_pipeline_command(CONFIG_PATH)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
