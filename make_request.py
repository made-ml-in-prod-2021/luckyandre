import numpy as np
import requests

from tests.synthetic_data_generator import synthetic_numeric_data_generator


if __name__ == "__main__":
    data = synthetic_numeric_data_generator(rand_state=7, size=10).drop(columns=['target'])
    request_features = list(data.columns)
    for i in range(10):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        response = requests.get(
            "http://127.0.0.1:8000/predict",
            json={
                "numerical_features": request_features,
                "data": [request_data]
            },
        )
        print(response.status_code)
        print(response.json())