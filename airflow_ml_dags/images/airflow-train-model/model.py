import os
import pickle

import pandas as pd
import click
from sklearn.ensemble import RandomForestClassifier


@click.command()
@click.option("--data_dir")
@click.option("--model_dir")
def train_model(data_dir: str, model_dir: str):
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    model = RandomForestClassifier()
    os.makedirs(model_dir, exist_ok=True)
    model.fit(train_df.drop(columns=['target']), train_df['target'])
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train_model()