from pydantic import BaseModel
from typing import Literal

class ClassifyRequest(BaseModel):
    image_b64: str
    variant: Literal["default", "hubble", "dirichlet"] = "default"

class predictions(BaseModel):
    date: str
    variant: str
    class_name: str
    probability: float
    image_b64: str