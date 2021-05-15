from dataclasses import dataclass


@dataclass()
class InferenceParams:
    source_data_path: str
    result_data_path: str
