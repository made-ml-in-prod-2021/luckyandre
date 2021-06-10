import os
import pandas as pd
import click


@click.command()
@click.option("--input_dir")
@click.option("--output_dir")
def preprocess(input_dir: str, output_dir):
    # read
    x = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))

    # processing
    x['target'] = y['target']

    # store
    os.makedirs(output_dir, exist_ok=True)
    x.to_csv(os.path.join(output_dir, "preprocessed_data.csv"))


if __name__ == '__main__':
    preprocess()