from fastapi import FastAPI
from .routers import router


app = FastAPI()
app.include_router(router, prefix="/predict")

@app.get("/")
async def root():
    return {"message": "welcome!"}

