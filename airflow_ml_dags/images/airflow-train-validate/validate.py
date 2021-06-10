import os
import pickle
import json

import pandas as pd
import click
from sklearn.metrics import roc_auc_score, accuracy_score


@click.command()
@click.option("--model_dir")
@click.option("--data_dir")
def validate(model_dir: str, data_dir: str):
    # model
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
       model = pickle.load(f)

    # data
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    # prediction
    pred = model.predict(test_df.drop(columns=['target']))

    # score
    score = {'roc_auc': roc_auc_score(test_df['target'], pred),
             'accuracy': accuracy_score(test_df['target'], pred)}
    with open(os.path.join(model_dir, 'score.json'), 'w') as f:
        json.dump(score, f)


if __name__ == '__main__':
    validate()