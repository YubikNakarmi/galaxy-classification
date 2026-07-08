from fastapi import APIRouter
from celery.result import AsyncResult
from .tasks import app
from .tasks import classify_image

router = APIRouter()