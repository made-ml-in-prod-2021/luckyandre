from py._path.local import LocalPath

import pytest
from typing import List, Optional


@pytest.fixture()
def random_state():
    return 7


@pytest.fixture()
def dataset_size():
    return 300


@pytest.fixture()
def dataset_path(tmpdir: LocalPath):
    return tmpdir.join("train.csv")


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def categorical_features_yes() -> List[str]:
    return ["categorical"]


@pytest.fixture()
def categorical_features_no() -> Optional[str]:
    return None


@pytest.fixture
def numerical_features_yes() -> List[str]:
    return [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]


@pytest.fixture()
def numerical_features_no() -> Optional[str]:
    return None


@pytest.fixture()
def features_to_drop_no() -> Optional[str]:
    return None
