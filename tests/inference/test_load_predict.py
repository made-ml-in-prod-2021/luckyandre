from typing import List
from py._path.local import LocalPath

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from tests.synthetic_data_generator import synthetic_numeric_data_generator
from ml_project.models import serialize_model
from ml_project.features import (
    Features_transformer,
    serialize_features_transformer
)
from ml_project.enities import (
    Params,
    FeatureParams,
    InferenceParams,
)
from ml_project.inference import (
    load_transformer,
    load_model,
    save_predicts,
)


@pytest.fixture
def params(tmpdir: LocalPath, numerical_features_yes: List[str]):

    expected_transformer_path = tmpdir.join("transformer.pkl")
    expected_model_path = tmpdir.join("model.pkl")
    expected_result_data_path = tmpdir.join("result.csv")

    params = Params(
        report_path="",
        train_data_path="",
        model_path=expected_model_path,
        features_transformer_path=expected_transformer_path,
        metric_path="",
        splitting_params="",
        train_params="",
        feature_params=FeatureParams(
            categorical_features="",
            numerical_features=numerical_features_yes, # to create simple transformer
            features_to_drop="",
            target_col=""),
        inference_params=InferenceParams(
            source_data_path="",
            result_data_path=expected_result_data_path
        )
    )
    return params


def test_load_transformer(params: Params):
    transformer = Features_transformer(params.feature_params)
    serialize_features_transformer(transformer, params.features_transformer_path)
    transformer = load_transformer(params.features_transformer_path)
    assert isinstance(transformer, Features_transformer)


def test_load_model(params: Params):
    model = RandomForestClassifier()
    serialize_model(model, params.model_path)
    model = load_model(params.model_path)
    assert isinstance(model, RandomForestClassifier)


def test_save_predicts(params: Params, random_state: int, dataset_size: int):
    data = synthetic_numeric_data_generator(random_state, dataset_size).drop(columns=['target'])
    predicts = np.random.randint(low=0, high=2, size=dataset_size)
    data_path = save_predicts(data, predicts, params.inference_params.result_data_path)
    assert data_path == params.inference_params.result_data_path