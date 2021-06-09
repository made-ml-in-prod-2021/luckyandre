import os

import click

from ml_project.synthetic_data_generator import synthetic_numeric_data_generator


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
    x.to_csv(os.path.join(output_dir, "data.csv"))
    y.to_csv(os.path.join(output_dir, "target.csv"))


if __name__ == '__main__':
    generate()