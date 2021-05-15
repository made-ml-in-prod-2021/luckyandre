from typing import List
from py._path.local import LocalPath

import pytest

from tests.synthetic_data_generator import synthetic_numeric_data_generator
from ml_project.train_pipeline import train_pipeline
from ml_project.inference_pipeline import inference_pipeline
from ml_project.enities import (
    Params,
    SplittingParams,
    TrainingParams,
    FeatureParams,
    InferenceParams,
)


@pytest.fixture
def params(tmpdir: LocalPath, numerical_features_yes: List[str], target_col: str):

    expected_train_data_path = tmpdir.join("train.csv")
    expected_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    expected_transformer_path = tmpdir.join("transformer.pkl")
    expected_source_data_path = tmpdir.join("source.csv")
    expected_result_data_path = tmpdir.join("result.csv")

    params = Params(
        report_path="",
        train_data_path=expected_train_data_path,
        model_path=expected_model_path,
        features_transformer_path=expected_transformer_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
        train_params=TrainingParams(model_type="RandomForestClassifier"),
        feature_params=FeatureParams(
            categorical_features=None,
            numerical_features=numerical_features_yes,
            features_to_drop=None,
            target_col=target_col),
        inference_params=InferenceParams(
            source_data_path=expected_source_data_path,
            result_data_path=expected_result_data_path
        )
    )
    return params


def test_inference_pipeline(random_state: int, dataset_size: int, params: Params):
    # data generation for train
    synthetic_data = synthetic_numeric_data_generator(random_state, dataset_size)
    synthetic_data.to_csv(params.train_data_path, index=False)

    # train
    _, _, _, _ = train_pipeline(params)

    # data generation for inference
    synthetic_data.drop(columns=['target'], inplace=True)
    synthetic_data.to_csv(params.inference_params.source_data_path, index=False)

    # inference
    path_to_predics, predicts = inference_pipeline(params)
    assert path_to_predics == params.inference_params.result_data_path
    assert len(predicts) == len(synthetic_data)