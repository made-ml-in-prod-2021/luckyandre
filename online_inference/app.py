import logging
import os
from typing import List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

from ml_project.features import Features_transformer, make_features
from ml_project.inference import load_model, load_transformer


logger = logging.getLogger(__name__)
app = FastAPI()


transformer: Optional[Features_transformer] = None
SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]
model: Optional[SklearnClassificationModel] = None


class Data_features(BaseModel):
    numerical_features: List[str]
    data: List[conlist(Union[float, int, None], min_items=13, max_items=13)]


def inference_pipeline(data: pd.DataFrame,
                       transformer: Features_transformer,
                       model: SklearnClassificationModel
):
    data_features = make_features(transformer, data)
    data['predicted_class'] = model.predict(data_features)
    return data[['predicted_class']].to_json(orient='index')


@app.on_event("startup")
def load_model_and_transformer():
    global model, transformer
    transformer_path = os.getenv("FEATURES_TRANSFORMER_PATH")
    model_path = os.getenv("MODEL_PATH")

    if transformer_path is None:
        err = f"FEATURES_TRANSFORMER_PATH {transformer_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    if model_path is None:
        err = f"MODEL_PATH {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_model(model_path)
    transformer = load_transformer(transformer_path)


@app.get("/")
def main():
    return "it is entry point for predictor"


@app.get("/predict")
def predict(features: Data_features):
    try:
        df = pd.DataFrame(data=features.data, columns=features.numerical_features)
    except:
        raise HTTPException(status_code=400, detail=f"Could not create data frame from provided data and columns")
    return inference_pipeline(df, transformer, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
