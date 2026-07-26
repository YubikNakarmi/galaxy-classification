import base64
from git import db
from sqlalchemy.orm import Session
from core.database import SessionLocal
from celery import Celery
from typing import Optional
from core.storage import get_storage
import uuid
import datetime as dt
import httpx
from classify.models import ClassificationResponseRecord, ImageUpload, Prediction
from exceptions import BentoResponseMismatchError, BentoServiceBadRequestError, BentoServiceTimeoutError
from prometheus_client import Counter, Histogram

PIPELINE_EXCEPTIONS = Counter(
    "pipeline_errors_total",
    "Total pipeline errors by type",
    ["error_code", "endpoint"],
)



celery_app = Celery('api', broker='redis://redis:6379/0',backend='redis://redis:6379/1')#Host redis on docker compose first
celery_app.conf.task_track_started = True


BENTOML_ROUTES = {#preset routes in bentoml/services.py
    "default": "http://bentoml:3000/default",
    "hubble": "http://bentoml:3000/hubble",
    "dirichlet": "http://bentoml:3000/dirichlet",
}


@celery_app.task(bind=True)#cleery queue task to classify image
def classify_image(self,image_bytes_b64:list[str],variant:str="default",response_id: Optional[str]=None):

    if variant not in BENTOML_ROUTES:
        raise ValueError(f"Unknown variant: {variant}")

    variant_url = BENTOML_ROUTES[variant]
    db: Session = SessionLocal() #init db


# tuple of images with fromat (images,(filename, file_bytes, content_type)) 
# that converts to multipart/form-data for httpx.post
    try:
        response_row = db.query(ClassificationResponseRecord).get(response_id)#filter pk of response record
        if response_row is None:
            raise ValueError(f"No ClassificationResponseRecord found for id {response_id}")

        # BentoML's route only accepts list[Image.Image], in format of [("images", (filename, file_bytes, content_type)),images,(...))]
        files = [
            ("images", (f"img{i}.jpg", base64.b64decode(b), "image/jpeg"))
            for i, b in enumerate(image_bytes_b64)
        ]

        try:
            
            http_response = httpx.post(variant_url, files=files, timeout=30)
            http_response.raise_for_status()

        except httpx.TimeoutException as exc:

            PIPELINE_EXCEPTIONS.labels(
                error_code="bento_timeout", endpoint=f"/{variant}"
            ).inc()

            raise BentoServiceTimeoutError(
            f"BentoML request timed out after 30s: {exc}",
            variant=variant,
            task_id=self.request.id)
        
        except httpx.HTTPStatusError as exc:

            if exc.response.status_code >= 500:
                raise self.retry(exc=exc)
            
            
            response_row.status = "failed"
            response_row.error_message = str(exc)
            db.commit()

            PIPELINE_EXCEPTIONS.labels(
                error_code="bento_bad_request", endpoint=f"/{variant}"
            ).inc()

            raise BentoServiceBadRequestError(
            f"BentoML rejected request with {exc.response.status_code}: {exc}",
            variant=variant,
            task_id=self.request.id,
            status_code=exc.response.status_code,
                )

        results_data = http_response.json()  # list[dict], same order as image_bytes_b64 

        # query batch id from upstream response_row to get all uploads for this batch
        uploads = (
            db.query(ImageUpload)
            .filter(ImageUpload.batch_id == response_row.batch_id)
            .order_by(ImageUpload.uploaded_at)
            .all()
        )

        # Ensure the number of uploads matches the number of results from BentoML
        if len(uploads) != len(results_data):
            response_row.status = "failed"
            response_row.error_message = (
                f"Mismatch: {len(uploads)} uploads vs {len(results_data)} BentoML results"
            )
            db.commit()

            PIPELINE_EXCEPTIONS.labels( #prometheus
                            error_code="bento_response_mismatch", endpoint=f"/{variant}"
                        ).inc()

            raise BentoResponseMismatchError(
                response_row.error_message,
                task_id=self.request.id,
                batch_id=response_row.batch_id,
                expected=len(uploads),
                received=len(results_data),)
                        
        
        storage = get_storage() #az blob client

        #for each prediction of images
        for upload, item in zip(uploads, results_data):
            heatmap_b64 = item.get("heatmap_b64")
            heatmap_blob_key = None
            heatmap_url = None

            if heatmap_b64:
                heatmap_bytes = base64.b64decode(heatmap_b64)
                heatmap_blob_key = f"heatmap-{uuid.uuid4()}.png"
                heatmap_url = storage.upload_blob(heatmap_blob_key, heatmap_bytes) # upload heatmap to azure blob storage

            prediction = Prediction(
                response_id=response_row.id,
                image_upload_id=upload.id,
                object_type=item.get("object_type"),
                confidence=item.get("confidence_scores"),
                heatmap_blob_key=heatmap_blob_key,
                heatmap_url=heatmap_url,
                processing_time_ms=int(item.get("heatmap_latency", 0) * 1000),
            )
            db.add(prediction)
            upload.status = "classified" #record status for each image upload
            db.commit()

        response_row.status = "completed"
        response_row.completed_at = dt.datetime.now(dt.timezone.utc)
        db.commit()

    finally:
        db.close()

    return results_data
 




