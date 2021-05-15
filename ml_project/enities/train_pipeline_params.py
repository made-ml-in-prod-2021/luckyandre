import yaml
from marshmallow_dataclass import class_schema
from dataclasses import dataclass

from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams
from .inference_params import InferenceParams


@dataclass()
class Params:
    report_path: str
    train_data_path: str
    model_path: str
    features_transformer_path: str
    metric_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    feature_params: FeatureParams
    inference_params: InferenceParams


ParamsSchema = class_schema(Params)


def read_params(path: str) -> Params:
    with open(path, "r") as input_stream:
        schema = ParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
