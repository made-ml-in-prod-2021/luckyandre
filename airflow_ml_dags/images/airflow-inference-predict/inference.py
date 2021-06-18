import os
import pickle

import pandas as pd
import click


@click.command()
@click.option("--model_dir")
@click.option("--output_data_dir")
@click.option("--input_data_dir")
def inference(input_data_dir: str, output_data_dir: str, model_dir: str):
    # model
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
       model = pickle.load(f)

    # data
    data_df = pd.read_csv(os.path.join(input_data_dir, 'inference_data.csv'))

    # prediction
    data_df['prediction'] = model.predict(data_df)

    # store
    os.makedirs(output_data_dir, exist_ok=True)
    data_df[['prediction']].to_csv(os.path.join(output_data_dir, "predicted_data.csv"), index=False)


if __name__ == '__main__':
    inference()