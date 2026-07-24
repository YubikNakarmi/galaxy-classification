import base64
import binascii
import uuid
from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, HTTPException
from celery.result import AsyncResult

from api.classify.models import ClassificationResponseRecord, ImageUpload
from api.tasks import celery_app, classify_image
from schemas import ClassifyRequest,ClassificationResponse,VARIANTS

from exceptions import *
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

API_EXCEPTIONS = Counter(
    "api_exceptions_total",
    "Total number of API exceptions",
    ["error_code", "endpoint"]
)

@router.post("/classify",response_model=ClassificationResponse)
async def classify_image_endpoint(
    request: ClassifyRequest, db: Session = Depends(get_db) #dep inject
    ):

    if request.variant not in VARIANTS:
        API_EXCEPTIONS.labels(error_code="invalid_variant", endpoint="/classify").inc()
        raise UnknownVariantError(message="Invalid variant specified.")

    if request.image_b64 is None or len(request.image_b64) == 0:
        API_EXCEPTIONS.labels(error_code="empty_image_list", endpoint="/classify").inc()
        raise EmptyImageListError(message="No images were provided in the request.")
    
    try:
        image_bytes = [base64.b64decode(img, validate=True) for img in request.image_b64] # for blob
    except binascii.Error:
        API_EXCEPTIONS.labels(error_code="invalid_base64", endpoint="/classify").inc()
        raise InvalidBase64ImageError(message = "One or more images failed to decode from base64.")

    except Exception as e:
        API_EXCEPTIONS.labels(error_code="unknown_error", endpoint="/classify").inc()
        raise PipelineError(message=f"An unexpected error occurred: {str(e)}")



    storage = get_storage() # init azure storage instance
    batch_id = str(uuid.uuid4()) #single request id

    urls = []
    blob_keys = []

    for i, request_image in enumerate(image_bytes):
        blob_key = f"{batch_id}-{i}.jpg"  # Generate a unique key for the blob, with multi images
        blob_keys.append(blob_key)

        blob_url = await run_in_threadpool(storage.upload_blob, blob_key, request_image)  # Upload image to Azure Blob Storage in async
        urls.append(blob_url)

    if None in urls:
        API_EXCEPTIONS.labels(error_code="blob_upload_failed", endpoint="/classify").inc()
        raise BlobUploadError(message="Failed to upload one or more images to storage.", batch_id=batch_id)

    uploads: list[ImageUpload] = []

    for blob_key, img_bytes in zip(blob_keys, image_bytes):#multi rows for multi images, with unique key but same batch id for tracking
        upload = ImageUpload(
            batch_id=batch_id,
            blob_key=blob_key,
            filename="upload.jpg",
            file_size=len(img_bytes),
            status="uploaded",
        )
        db.add(upload)
        uploads.append(upload)
  
    db.commit()
    for upload in uploads:
        db.refresh(upload)

    response_row = ClassificationResponseRecord(
        batch_id=batch_id,
        variant=request.variant,
        status="processing",)

    db.add(response_row)

    db.commit()
    db.refresh(response_row)

    task = classify_image.delay(request.image_b64, request.variant,response_row.id) #call celery

    response_row.task_id = task.id
    db.commit()


    return {
        "batch_id": batch_id,
        "task_id": task.id,
        "image_upload_ids": [u.id for u in uploads],
        "urls": urls,
        "status": "processing",
    }#returns the task id to the client for tracking from celery


@router.get("/results/{batch_id}")
async def get_classification_results(batch_id: str, db: Session = Depends(get_db)):
    pass


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