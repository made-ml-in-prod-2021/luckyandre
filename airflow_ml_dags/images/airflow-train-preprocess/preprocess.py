import os
import pandas as pd
import click


@click.command()
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--mode")
def preprocess(input_dir: str, output_dir: str, mode: str):
    # read
    x = pd.read_csv(os.path.join(input_dir, "data.csv"))

    # processing
    pass

    # store
    os.makedirs(output_dir, exist_ok=True)
    if mode == 'train':
        y = pd.read_csv(os.path.join(input_dir, "target.csv"))
        x['target'] = y['target']
        x.to_csv(os.path.join(output_dir, "preprocessed_data.csv"), index=False)
    elif mode == 'inference':
        x.to_csv(os.path.join(output_dir, "inference_data.csv"), index=False)


if __name__ == '__main__':
    preprocess()