import os

import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command()
@click.option("--dir")
def split(dir: str):
    data = pd.read_csv(os.path.join(dir, "preprocessed_data.csv"))
    x_train, x_test = train_test_split(data, test_size=0.3, random_state=7)
    x_train.to_csv(os.path.join(dir, "train_data.csv"))
    x_test.to_csv(os.path.join(dir, "test_data.csv"))


if __name__ == '__main__':
    split()