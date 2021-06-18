import os

import click
import numpy as np
import pandas as pd


def synthetic_numeric_data_generator(rand_state: int = 7, size: int = 1000) -> pd.DataFrame:
    np.random.seed(rand_state)
    df = pd.DataFrame()
    df["age"] = np.random.normal(loc=50, scale=10, size=size)
    df["sex"] = np.random.randint(low=0, high=2, size=size)
    df["cp"] = np.random.randint(low=0, high=4, size=size)
    df["trestbps"] = np.random.normal(loc=130, scale=15, size=size)
    df["chol"] = np.random.normal(loc=250, scale=50, size=size)
    df["fbs"] = np.random.randint(low=0, high=2, size=size)
    df["restecg"] = np.random.randint(low=0, high=3, size=size)
    df["thalach"] = np.random.normal(loc=150, scale=20, size=size)
    df["exang"] = np.random.randint(low=0, high=2, size=size)
    df["oldpeak"] = np.random.exponential(scale=0.9, size=size)
    df["slope"] = np.random.randint(low=0, high=3, size=size)
    df["ca"] = np.random.randint(low=0, high=5, size=size)
    df["thal"] = np.random.randint(low=0, high=4, size=size)
    df["target"] = np.random.randint(low=0, high=2, size=size)
    df = df.astype(int)
    df["oldpeak"] = np.random.exponential(scale=0.9, size=size) # the only float column
    return df


@click.command()
@click.option("--size", default=1000)
@click.option("--random_state", default=7)
@click.option("--output_dir")
def generate(output_dir: str, random_state: int, size: int):
    # data
    df = synthetic_numeric_data_generator(random_state, size)
    y = df['target']
    x = df.drop(columns=['target'])

    # save
    os.makedirs(output_dir, exist_ok=True)
    x.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    generate()