from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: Optional[str]
    numerical_features: List[str]
    features_to_drop: Optional[str]
    target_col: str
