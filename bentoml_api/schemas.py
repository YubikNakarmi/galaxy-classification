from typing import Optional
from typing_extensions import TypedDict


class ClassificationResult(TypedDict):
    object_type: str
    confidence_scores: dict[str, float]
    heatmap_b64: Optional[str]


class HubbleResult(TypedDict):
    object_type: str
    confidence_scores: dict[str, float]


class DirichletResult(TypedDict):
    object_type: str
    confidence_scores: dict[str, float]
    uncertainty: float
    alphas: dict[str, float]