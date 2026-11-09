from contextlib import asynccontextmanager

from fastapi import FastAPI
from api.classify.routers import router
from core.database import init_db
from core.storage import get_storage, close_storage

from prometheus_client import make_asgi_app
from src.utils.logging_config import setup_app_logging, intercept_stdlib_logging
from src.utils.yaml_utils import load_yaml

core_config = load_yaml("configs/core.yaml")
setup_app_logging(config=core_config, service="api", json_logs=False)
intercept_stdlib_logging()  # Redirect stdlib logging to loguru


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db() 
    get_storage()   # force connection at startup
    yield
    close_storage()#shutdown

app = FastAPI(title="Galaxy Classification API", 
              version="0.6.0", 
              lifespan=lifespan,)



app.include_router(router,prefix="/classification",tags=["classification"])

metrics = make_asgi_app() #for prometheus metrics
app.mount("/metrics", metrics)

@app.get("/")
async def root():
    return {"message": "welcome!"}

@app.get("/health")
def health():
    return {"status": "ok"}
