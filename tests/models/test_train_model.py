import os
import pickle
from typing import List, Optional, Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestClassifier

from ml_project.enities import TrainingParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.features.build_features import make_features, extract_target, Features_transformer
from ml_project.models.model_fit_predict import train_model, serialize_model
from tests.synthetic_data_generator import synthetic_numeric_data_generator


@pytest.fixture
def features_and_target(
        categorical_features_no: Optional[str],
        numerical_features_yes: List[str],
        features_to_drop_no: Optional[str],
        target_col: str,
        random_state: int,
        dataset_size: int
) -> Tuple[pd.DataFrame, pd.Series]:

    params = FeatureParams(
        categorical_features=categorical_features_no,
        numerical_features=numerical_features_yes,
        features_to_drop=features_to_drop_no,
        target_col=target_col,
    )

    data = synthetic_numeric_data_generator(random_state, dataset_size)
    transformer = Features_transformer(params)
    transformer.fit(data.drop(columns=[params.target_col]))
    features = make_features(transformer, data.drop(columns=[params.target_col]))
    target = extract_target(data, params)

    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists(real_output)
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)