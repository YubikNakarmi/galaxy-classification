from pydantic import BaseModel
from typing import Literal

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class ClassifyRequest(BaseModel):
    image_b64: str
    variant: Literal["default", "hubble", "dirichlet"] = "default"




class ClassificationResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    image_upload_id: str
    task_id: str
    object_type: Optional[str]
    confidence: Optional[float]
    confidence_scores: Optional[dict]
    heatmap_url: Optional[str]
    processing_time_ms: Optional[int]
    variant: str
    created_at: datetime

class ClassificationResponse(BaseModel):

    task_id: str
    image_upload_id: str
    blob_url: str