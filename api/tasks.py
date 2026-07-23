import base64

from celery import Celery
import httpx

celery_app = Celery('api', broker='redis://redis:6379/0',backend='redis://redis:6379/1')#Host redis on docker compose first
celery_app.conf.task_track_started = True


BENTOML_ROUTES = {#preset routes in bentoml/services.py
    "default": "http://bentoml:3000/default",
    "hubble": "http://bentoml:3000/hubble",
    "dirichlet": "http://bentoml:3000/dirichlet",
}


@celery_app.task(bind=True)#cleery queue task to classify image
def classify_image(self,image_bytes_b64:list[str],variant:str="default"):

    variant_url = BENTOML_ROUTES[variant]

# tuple of images with fromat (images,(filename, file_bytes, content_type)) that converts to multipart/form-data for httpx.post

    files = [
    ("images", (f"img{i}.jpg", base64.b64decode(b), "image/jpeg"))
    for i, b in enumerate(image_bytes_b64) ]

    response = httpx.post(variant_url, files=files, timeout=30)
    response.raise_for_status()
    
    return response.json()
   


