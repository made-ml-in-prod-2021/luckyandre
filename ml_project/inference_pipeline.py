import logging
import sys
import click

from ml_project.data import read_data
from ml_project.features import make_features
from ml_project.models import predict_model
from ml_project.enities import (
    Params,
    read_params
)
from ml_project.inference import (
    load_transformer,
    load_model,
    save_predicts,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def inference_pipeline(inference_pipeline_params: Params):
    # train, val data
    logger.info(f"start inference pipeline with params {inference_pipeline_params.inference_params}")
    data = read_data(inference_pipeline_params.inference_params.source_data_path)
    logger.info(f"data.shape is {data.shape}")

    # features extraction
    transformer = load_transformer(inference_pipeline_params.features_transformer_path)
    data_features = make_features(transformer, data)
    logger.info(f"data_features.shape is {data.shape}")

    # predict
    model = load_model(inference_pipeline_params.model_path)
    predicts = predict_model(model, data_features)
    logger.info(f"predicts shape is {predicts.shape}")

    # save
    path_to_predics = save_predicts(data, predicts, inference_pipeline_params.inference_params.result_data_path)
    logger.info(f"predicted data was saved")

    return path_to_predics, predicts


@click.command(name="inference_pipeline")
@click.argument("config_path")
def inference_pipeline_command(config_path: str):
    params = read_params(config_path)
    inference_pipeline(params)


if __name__ == "__main__":
    inference_pipeline_command()
