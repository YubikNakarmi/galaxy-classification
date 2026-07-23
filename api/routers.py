import base64
import binascii
import uuid
from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, HTTPException
from celery.result import AsyncResult

from api.models import ClassificationResult, ImageUpload

from .tasks import celery_app
from .tasks import classify_image
from .schemas import ClassifyRequest,ClassificationResponse
from core.database import get_db
from sqlalchemy.orm import Session
from core.storage import get_storage
from celery.result import AsyncResult
from loguru import logger
from src.utils.logging_config import setup_app_logging,intercept_stdlib_logging
from src.utils.yaml_utils import load_yaml
from pathlib import Path
from prometheus_client import Counter, Histogram

root = Path(__file__).parent.parent
cfg = load_yaml(root / 'configs' / 'core.yaml')
setup_app_logging(config = cfg,service = "API",json_logs = True)
intercept_stdlib_logging() #stdlib logging for container

router = APIRouter()

@router.post("/classify",response_model=ClassificationResponse)

async def classify_image_endpoint(
    request: ClassifyRequest, db: Session = Depends(get_db) #dep inject
    ):

    try:
        image_bytes = base64.b64decode(request.image_b64, validate=True)
    except binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")


    storage = get_storage() # init azure storage instance
    blob_key = f"{uuid.uuid4()}.jpg"  # Generate a unique key for the blob
    url = await run_in_threadpool(storage.upload_blob, blob_key, request.image_b64)  # Upload image to Azure Blob Storage in async

    if url is None:
        raise HTTPException(status_code=500, detail="Failed to upload image to Azure Blob Storage")

    upload = ImageUpload(
        blob_key=blob_key,
        filename="upload.jpg",
        file_size=len(image_bytes),
        status="uploaded",
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)


    task = classify_image.delay(request.image_b64, request.variant) #call celery

    result = ClassificationResult(
        image_upload_id=upload.id,
        task_id=task.id,
        variant=request.variant,
    )
    db.add(result)
    db.commit()

    return {
        "task_id": task.id,
        "image_upload_id": upload.id,
        "blob_url": url,
    }#returns the task id to the client for tracking from celery





@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):

    result = AsyncResult(task_id, app=celery_app)#check result form celery

    
    if result.state == "PENDING":
        return {"status": "pending"}
    if result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}
    if result.ready():
        return {"status": "done", "result": result.result}
    return {"status": result.state.lower()}