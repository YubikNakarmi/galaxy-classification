import base64
import os
from io import BytesIO
from sqlalchemy.orm import Session
from core.database import SessionLocal
from celery import Celery
from celery.exceptions import Retry
from typing import Optional
from core.storage import get_storage
import uuid
import datetime as dt
import httpx
from PIL import Image, UnidentifiedImageError
from .classify.models import ClassificationResponseRecord, ImageUpload, Prediction
from .exceptions import (
    BentoResponseMismatchError,
    BentoServiceBadRequestError,
    BentoServiceTimeoutError,
    BentoServiceUnavailableError,
)
from prometheus_client import Counter
from loguru import logger
from core.settings import settings

PIPELINE_EXCEPTIONS = Counter(
    "pipeline_errors_total",
    "Total pipeline errors by type",
    ["error_code", "endpoint"],
)



celery_app = Celery(
    "api",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
)
celery_app.conf.update(
    task_track_started=True,
    worker_pool="solo" if os.name == "nt" else "prefork", #for linux/windows
    worker_concurrency=1 if os.name == "nt" else None,
)


def _mark_response_failed(response_id: Optional[str], error_message: str) -> None:
    if response_id is None:
        return

    db = SessionLocal()
    try:
        response_row = db.get(ClassificationResponseRecord, response_id)
        if response_row is not None:
            response_row.status = "failed"
            response_row.error_message = error_message[:1024]
            db.commit()
    finally:
        db.close()


BENTOML_ROUTES = {#preset routes in bentoml/services.py
    "default": os.getenv("BENTOML_DEFAULT_URL", "http://bentoml:3000/default"),
    "hubble": os.getenv("BENTOML_HUBBLE_URL", "http://bentoml:3000/hubble"),
    "dirichlet": os.getenv("BENTOML_DIRICHLET_URL", "http://bentoml:3000/dirichlet"),
}


@celery_app.task(bind=True)#cleery queue task to classify image
def classify_image(self,image_bytes_b64:list[str],variant:str="default",response_id: Optional[str]=None):

    if variant not in BENTOML_ROUTES:
        logger.exception(f"Unknown variant: {variant}")
        raise ValueError(f"Unknown variant: {variant}")

    variant_url = BENTOML_ROUTES[variant]
    db: Session = SessionLocal() #init db


# tuple of images with fromat (images,(filename, file_bytes, content_type)) 
# that converts to multipart/form-data for httpx.post
    try:
        response_row = db.query(ClassificationResponseRecord).get(response_id)#filter pk of response record
        if response_row is None:
            logger.exception(f"No ClassificationResponseRecord found for id {response_id}")
            raise ValueError(f"No ClassificationResponseRecord found for id {response_id}")

        # Bentoml route only accepts list[Image.Image], 
        # in format of [("images", (filename, file_bytes, content_type)),images,(...))]
        files = []
        for i, encoded_image in enumerate(image_bytes_b64):
            image_bytes = base64.b64decode(encoded_image, validate=True)
            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    image_format = image.format
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Image {i} is not a valid image: {exc}") from exc

            if image_format is None:
                raise ValueError(f"Image {i} has no detectable image format")

            image_format = image_format.lower()
            extension = "jpg" if image_format == "jpeg" else image_format #correct format for bentoml responsew hich takes imgs
            content_type = f"image/{'jpeg' if image_format == 'jpeg' else image_format}"
            files.append(
                (
                    "images",
                    (f"img{i}.{extension}", image_bytes, content_type),
                )
            )

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
                response_body = exc.response.text[:1024]
                logger.error(
                    "BentoML returned HTTP {} for {}: {}",
                    exc.response.status_code,
                    variant_url,
                    response_body,
                )
                if self.request.retries >= 3: #mroe then 3 retries fail
                    error_message = (
                        f"BentoML returned HTTP {exc.response.status_code}: "
                        f"{response_body}"
                    )
                    response_row.status = "failed"
                    response_row.error_message = error_message
                    db.commit()
                    raise BentoServiceUnavailableError(
                        error_message,
                        variant=variant,
                        task_id=self.request.id,
                        status_code=exc.response.status_code,
                    ) from exc
                raise self.retry(exc=exc, countdown=30, max_retries=3)
            
            
            response_row.status = "failed"
            response_row.error_message = str(exc)
            logger.exception(f"Error occurred while processing request: {exc}")
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
        except Retry:
            raise
        except Exception as exc:
            response_row.status = "failed"
            response_row.error_message = str(exc)
            logger.exception(f"Unexpected error occurred while processing request: {exc}")
            db.commit()

            PIPELINE_EXCEPTIONS.labels(
                error_code="bento_unexpected_error", endpoint=f"/{variant}"
            ).inc()

            raise 

        results_data = http_response.json()  # list[dict], same order as image_bytes_b64 

        # query batch id from upstream response_row to get all uploads for this batch
        uploads = (
            db.query(ImageUpload)
            .filter(ImageUpload.batch_id == response_row.batch_id)
            .order_by(ImageUpload.uploaded_at)
            .all()
        )
        logger.info(f"Received {len(results_data)} results from BentoML for {len(uploads)} uploads")
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

            logger.exception(
                f"Mismatch: {len(uploads)} uploads vs {len(results_data)} BentoML results"
            )
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

    except Retry:
        raise
    except Exception as exc:
        _mark_response_failed(response_id, str(exc))
        raise
    finally:
        db.close()

    return results_data
 
