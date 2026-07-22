from contextlib import asynccontextmanager

from fastapi import FastAPI
from .routers import router
from core.database import init_db
from core.storage import get_storage, close_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db() 
    get_storage()   # force connection at startup
    yield
    close_storage()#shutdown

app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/predict")

@app.get("/")
async def root():
    return {"message": "welcome!"}

