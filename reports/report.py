import sys
import logging
import click

import pandas as pd
from pandas_profiling import ProfileReport

from ml_project.enities import (
    Params,
    read_params
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def make_report(params: Params):
    logger.info("EDA report preparation started")
    source_df = pd.read_csv(params.train_data_path)

    # report
    profile = ProfileReport(source_df)
    profile.to_file(output_file=params.report_path)
    logger.info("EDA report preparation completed")


@click.command(name="make_report")
@click.argument("config_path")
def make_report_command(config_path: str):
    params = read_params(config_path)
    make_report(params)


if __name__ == "__main__":
    make_report_command()