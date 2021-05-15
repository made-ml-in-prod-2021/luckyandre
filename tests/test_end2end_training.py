import os
from py._path.local import LocalPath

import pytest
from typing import Optional, List

from tests.synthetic_data_generator import synthetic_numeric_data_generator
from ml_project.train_pipeline import train_pipeline
from ml_project.enities import (
    Params,
    SplittingParams,
    FeatureParams,
    TrainingParams,
    InferenceParams,
)


@pytest.fixture()
def params(
    dataset_path: str,
    tmpdir: LocalPath,
    categorical_features_no: Optional[str],
    numerical_features_yes: List[str],
    target_col: str,
    features_to_drop_no: Optional[str],
):

    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    expected_features_transformer_path = tmpdir.join("features_transformer.pkl")

    params = Params(
        report_path="",
        train_data_path=dataset_path,
        model_path=expected_output_model_path,
        features_transformer_path=expected_features_transformer_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
        train_params=TrainingParams(model_type="RandomForestClassifier"),
        feature_params=FeatureParams(
            numerical_features=numerical_features_yes,
            categorical_features=categorical_features_no,
            target_col=target_col,
            features_to_drop=features_to_drop_no),
        inference_params=InferenceParams(source_data_path="", result_data_path="")
    )

    return params


def test_train_pipeline(dataset_path: str, random_state: int, dataset_size: int, params: Params):
    # data generation
    synthetic_data = synthetic_numeric_data_generator(random_state, dataset_size)
    synthetic_data.to_csv(dataset_path, index=False)

    # train_pipeline
    path_to_feature_transformer, path_to_model, path_to_metrics, metrics = train_pipeline(params)
    assert metrics["roc_auc"] > 0.5
    assert os.path.exists(path_to_feature_transformer)
    assert os.path.exists(path_to_model)
    assert os.path.exists(path_to_metrics)