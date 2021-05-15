import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ml_project.enities.feature_params import FeatureParams


class Features_transformer(object):

    def __init__(self, params: FeatureParams):
        self.params = params
        if (params.categorical_features is not None) and (params.numerical_features is not None):
            transformer = ColumnTransformer(
                [
                    (
                        "categorical_pipeline",
                        self.build_categorical_pipeline(),
                        params.categorical_features,
                    ),
                    (
                        "numerical_pipeline",
                        self.build_numerical_pipeline(),
                        params.numerical_features,
                    ),
                ]
            )

        elif params.categorical_features is None:
            self.params.categorical_features = []
            transformer = ColumnTransformer(
                [
                    (
                        "numerical_pipeline",
                        self.build_numerical_pipeline(),
                        params.numerical_features,
                    )
                ]
            )

        elif params.numerical_features is None:
            self.params.numerical_features = []
            transformer = ColumnTransformer(
                [
                    (
                        "categorical_pipeline",
                        self.build_categorical_pipeline(),
                        params.categorical_features,
                    )
                ]
            )
        else:
            raise NotImplementedError("Numerical and categorical columns is None")

        self.transformer = transformer

    @staticmethod
    def build_categorical_pipeline() -> Pipeline:
        categorical_pipeline = Pipeline(
            [
                ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
                ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent"))
            ]
        )
        return categorical_pipeline

    @staticmethod
    def build_numerical_pipeline() -> Pipeline:
        num_pipeline = Pipeline(
            [("impute", SimpleImputer(missing_values=np.nan, strategy="mean"))]
        )
        return num_pipeline

    def check_input_df(self, df: pd.DataFrame) -> pd.DataFrame:

        # check type of input
        if type(df) != pd.DataFrame:
            raise NotImplementedError("DataFrame object type expected as input")

        # check matching of columns with params
        params_categorical = self.params.categorical_features
        params_numerical = self.params.numerical_features
        param_cols = params_categorical + params_numerical
        param_cols.sort()
        df_cols = df.columns.to_list()
        df_cols.sort()
        if param_cols != df_cols:
            raise NotImplementedError(f"DataFrame columns don't match with params",
                                      f"Expected {param_cols}",
                                      f"Got {df_cols}")

        # check matching of columns types with params
        df_cols_numeric = df.select_dtypes(
            include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.to_list()
        df_cols_numeric.sort()
        params_numerical.sort()
        if df_cols_numeric != params_numerical:
            raise NotImplementedError("Numeric columns don't match with params")
        df_cols_categorical = [col for col in df.columns if col not in df_cols_numeric]
        df_cols_categorical.sort()
        params_categorical.sort()
        if df_cols_categorical != params_categorical:
            raise NotImplementedError("Categorical columns don't match with params")

        # fix columns order
        df = df[
            [n_col for n_col in self.params.numerical_features] + [c_col for c_col in self.params.categorical_features]]

        return df

    def fit(self, df: pd.DataFrame):
        df = self.check_input_df(df)
        self.transformer.fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.check_input_df(df)
        return pd.DataFrame(self.transformer.transform(df))

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.check_input_df(df)
        self.transformer.fit(df)
        return pd.DataFrame(self.transformer.transform(df))


def make_features(transformer: Features_transformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target


def serialize_features_transformer(transformer: Features_transformer, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(transformer, f)
    return output
