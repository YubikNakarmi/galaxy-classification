## `/classify`

Accepts one or more base64-encoded images and returns a batch classification task:

```json
{
 "batch_id": "batch-id",
 "task_id": "celery-task-id",
 "image_upload_ids": ["upload-id"],
 "urls": ["https://storage.example/image.jpg"],
 "status": "processing"
}
```