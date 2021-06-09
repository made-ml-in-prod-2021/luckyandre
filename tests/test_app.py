from unittest.mock import patch
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from online_inference.app import app
from ml_project.features import Features_transformer
from ml_project.entities import FeatureParams

from ml_project.synthetic_data_generator import synthetic_numeric_data_generator


client = TestClient(app)


# dummy global transformer and model
synthetic_data = synthetic_numeric_data_generator(rand_state=7, size=300)
model = RandomForestClassifier()
model.fit(synthetic_data.drop(columns=['target']), synthetic_data['target'])
transformer = Features_transformer(
    FeatureParams(
        numerical_features=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        categorical_features=None,
        features_to_drop=None,
        target_col=None
    )
)
transformer.fit(synthetic_data.drop(columns=['target']))


def test_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "it is entry point for predictor"


@patch("online_inference.app.model", model)
@patch("online_inference.app.transformer", transformer)
def test_predict():

    # dummy data
    synthetic_data = synthetic_numeric_data_generator(rand_state=7, size=1)
    synthetic_data.drop(columns=['target'], inplace=True)

    for i in range(1):
        request_features = list(synthetic_data.columns)
        request_data = [x.item() if isinstance(x, np.generic) else x for x in synthetic_data.iloc[i].tolist()]
        response = client.get(
            "/predict",
            json={
                "numerical_features": request_features,
                "data": [request_data]
            },
        )
    assert response.status_code == 200