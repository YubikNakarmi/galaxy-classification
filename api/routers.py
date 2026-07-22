from fastapi import APIRouter, Depends
from celery.result import AsyncResult
from .tasks import celery_app
from .tasks import classify_image
from .schemas import ClassifyRequest
from core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()

router.post("/classify")
async def classify_image_endpoint(
    request: ClassifyRequest, db: Session = Depends(get_db)
):
    task = classify_image.delay(request.image_b64, request.variant) #call celery
    return {"task_id": task.id}#returns the task id to the client for tracking from celery

router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):

    result = AsyncResult(task_id, app=celery_app)#check result form celery

    
    if result.state == "PENDING":
        return {"status": "pending"}
    if result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}
    if result.ready():
        return {"status": "done", "result": result.result}
    return {"status": result.state.lower()}