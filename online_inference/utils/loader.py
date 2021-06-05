import pickle
from typing import Union

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]  # used for function signature definition


def load_object(path: str) -> Union[ColumnTransformer, SklearnClassificationModel]:
    loaded_object = pickle.load(open(path, 'rb'))
    return loaded_object

