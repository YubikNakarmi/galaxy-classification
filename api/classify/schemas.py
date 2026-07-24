from pydantic import BaseModel
from typing import Literal

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class ClassifyRequest(BaseModel):
    image_b64: list[str]
    variant: Literal["default", "hubble", "dirichlet"] = "default" #enfore these variants to prevent downstream errors 


class ClassificationResponse(BaseModel):

    task_id: str
    image_upload_id: str
    blob_url: str