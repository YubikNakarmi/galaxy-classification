from celery import Celery

app = Celery('api', broker='redis://localhost:6379/0')
app.conf.update(backend='redis://localhost:6379/0', beat_schedule={
    'classify-image-every-10-seconds': {
        'task': 'api.queue.classify_image',
        'schedule': 10.0
    }
})


@app.task

def classify_image(image_path):
    # Placeholder for image classification logic
    # In a real implementation, you would load the image and run it through a model
    return f"Classified image at {image_path}"


