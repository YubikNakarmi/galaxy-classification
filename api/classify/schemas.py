from pydantic import BaseModel
from typing import Literal


from pydantic import BaseModel, ConfigDict

class ClassifyRequest(BaseModel):
    image_b64: list[str]
    variant: Literal["default", "hubble", "dirichlet"] = "default" #enfore these variants to prevent downstream errors 


class ClassificationResponse(BaseModel):

    task_id: str
    image_upload_id: str
    blob_url: str

VARIANTS = ["default", "hubble", "dirichlet"]