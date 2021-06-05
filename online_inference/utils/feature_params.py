from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None
    features_to_drop: Optional[List[str]] = None
