from numpy.testing import assert_allclose

from ml_project.data.make_dataset import read_data, split_train_val_data
from ml_project.enities import SplittingParams
from tests.synthetic_data_generator import synthetic_numeric_data_generator


def test_read_data(dataset_path: str, target_col: str, random_state: int, dataset_size: int):
    # data generation
    synthetic_data = synthetic_numeric_data_generator(random_state, dataset_size)
    synthetic_data.to_csv(dataset_path, index=False)

    # read data
    data = read_data(dataset_path)
    assert_allclose(synthetic_data['oldpeak'].to_numpy(), data['oldpeak'].to_numpy(), atol=0.001, verbose=True)
    assert synthetic_data.drop(columns=['oldpeak']).equals(data.drop(columns=['oldpeak']))
    assert target_col in data.keys()


def test_split_train_val_data(random_state: int, dataset_size: int):
    # data generation
    data = synthetic_numeric_data_generator(random_state, dataset_size)

    # data split
    val_size = 0.2
    splitting_params = SplittingParams(random_state=239, val_size=val_size)
    train, val = split_train_val_data(data, splitting_params)
    assert_allclose(train.shape[0], dataset_size * (1 - val_size))
    assert train.shape[1] == data.shape[1]
    assert_allclose(val.shape[0], dataset_size * val_size)
    assert val.shape[1] == data.shape[1]