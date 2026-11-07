from typing import Literal

from pydantic import BaseModel

class ClassifyRequest(BaseModel):
    image_b64: list[str]
    variant: Literal["default", "hubble", "dirichlet"] = "default" #enfore these variants to prevent downstream errors 


class ClassificationResponse(BaseModel):
    batch_id: str
    task_id: str
    image_upload_ids: list[str]
    urls: list[str]
    status: Literal["processing", "completed", "failed"]

VARIANTS = ["default", "hubble", "dirichlet"]
