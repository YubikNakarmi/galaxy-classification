from celery import Celery
import httpx

app = Celery('api', broker='redis://localhost:6379/0')
app.conf.update(backend='redis://localhost:6379/0', beat_schedule={
    'classify-image-every-10-seconds': {
        'task': 'api.queue.classify_image',
        'schedule': 10.0
    }
})
app.conf.task_track_started = True


BENTOML_ROUTES = {#preset routes in bentoml/services.py
    "default": "http://bentoml:3000/default",
    "hubble": "http://bentoml:3000/hubble",
    "dirichlet": "http://bentoml:3000/dirichlet",
}


@app.task(bind=True)
def classify_image(self,image_bytes_b64:str,variant:str="default"):

    variant_url = BENTOML_ROUTES[variant]

    response = httpx.post(variant_url, json = {"images":image_bytes_b64},timeout=30)
    response.raise_for_status()
    return response.json()
   


