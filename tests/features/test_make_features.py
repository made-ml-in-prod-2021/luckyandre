from typing import List, Optional

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ml_project.enities.feature_params import FeatureParams
from ml_project.features.build_features import make_features, extract_target, Features_transformer
from tests.synthetic_data_generator import (
    synthetic_numeric_data_generator,
    synthetic_categorical_data_generator,
    synthetic_numeric_and_categorical_data_generator
)


@pytest.fixture
def feature_params_numeric(
    categorical_features_no: Optional[str],
    numerical_features_yes: List[str],
    features_to_drop_no: Optional[str],
    target_col: str,
) -> FeatureParams:

    params = FeatureParams(
        categorical_features=categorical_features_no,
        numerical_features=numerical_features_yes,
        features_to_drop=features_to_drop_no,
        target_col=target_col,
    )

    return params


@pytest.fixture
def feature_params_categorical(
    categorical_features_yes: List[str],
    numerical_features_no: Optional[str],
    features_to_drop_no: Optional[str],
    target_col: str,
) -> FeatureParams:

    params = FeatureParams(
        categorical_features=categorical_features_yes,
        numerical_features=numerical_features_no,
        features_to_drop=features_to_drop_no,
        target_col=target_col,
    )

    return params


@pytest.fixture
def feature_params_numerical_categorical(
    categorical_features_yes: List[str],
    numerical_features_yes: List[str],
    features_to_drop_no: Optional[str],
    target_col: str,
) -> FeatureParams:

    params = FeatureParams(
        categorical_features=categorical_features_yes,
        numerical_features=numerical_features_yes,
        features_to_drop=features_to_drop_no,
        target_col=target_col,
    )

    return params


def test_make_features_numerical(random_state: int, dataset_size: int, feature_params_numeric: FeatureParams):
    # data generation
    data = synthetic_numeric_data_generator(random_state, dataset_size)

    # features processing
    transformer = Features_transformer(feature_params_numeric)
    transformer.fit(data.drop(columns=[feature_params_numeric.target_col]))
    features = make_features(transformer, data.drop(columns=[feature_params_numeric.target_col]))
    assert not pd.isnull(features).any().any()
    assert data.shape[0] == features.shape[0]
    assert data.shape[1] - 1 == features.shape[1] # reduced on target


def test_make_features_categorical(random_state: int, dataset_size: int, feature_params_categorical: FeatureParams):
    # data generation
    data = synthetic_categorical_data_generator(random_state, dataset_size)

    # features processing
    transformer = Features_transformer(feature_params_categorical)
    transformer.fit(data.drop(columns=[feature_params_categorical.target_col]))
    features = make_features(transformer, data.drop(columns=[feature_params_categorical.target_col]))
    assert not pd.isnull(features).any().any()
    assert data.shape[0] == features.shape[0]
    assert data.shape[1] - 1 == features.shape[1] # reduced on target


def test_make_features_numerical_categorical(
        random_state: int,
        dataset_size: int,
        feature_params_numerical_categorical: FeatureParams
):

    # data generation
    data = synthetic_numeric_and_categorical_data_generator(random_state, dataset_size)

    # features processing
    transformer = Features_transformer(feature_params_numerical_categorical)
    transformer.fit(data.drop(columns=[feature_params_numerical_categorical.target_col]))
    features = make_features(transformer, data.drop(columns=[feature_params_numerical_categorical.target_col]))
    assert not pd.isnull(features).any().any()
    assert data.shape[0] == features.shape[0]
    assert data.shape[1] - 1 == features.shape[1] # reduced on target


def test_extract_features(random_state: int, dataset_size: int, feature_params_numeric: FeatureParams):
    # data generation
    data = synthetic_numeric_data_generator(random_state, dataset_size)

    target = extract_target(data, feature_params_numeric)
    assert_allclose(data['target'].to_numpy(), target.to_numpy())