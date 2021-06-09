import numpy as np
import pandas as pd


def synthetic_numeric_data_generator(rand_state: int, size: int) -> pd.DataFrame:
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


def synthetic_categorical_data_generator(rand_state: int, size: int) -> pd.DataFrame:
    np.random.seed(rand_state)
    df = pd.DataFrame()
    df['categorical'] = [chr(i) for i in np.random.randint(low=ord('A'), high=ord('z') + 1, size=size)]
    df["target"] = np.random.randint(low=0, high=2, size=size)
    return df


def synthetic_numeric_and_categorical_data_generator(rand_state: int, size: int) -> pd.DataFrame:
    np.random.seed(rand_state)
    df = synthetic_numeric_data_generator(rand_state, size)
    df['categorical'] = [chr(i) for i in np.random.randint(low=ord('A'), high=ord('z') + 1, size=size)]
    return df