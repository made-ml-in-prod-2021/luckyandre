import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]  # used for function signature definition


def load_transformer(path: str) -> ColumnTransformer:
    transformer = pickle.load(open(path, 'rb'))
    return transformer


def load_model(path: str) -> SklearnClassificationModel:
    model = pickle.load(open(path, 'rb'))
    return model


def save_predicts(source: pd.DataFrame, predicts: np.array, output: str) -> str:
    source['prediction'] = predicts
    source.to_csv(output, index=False)
    return output
