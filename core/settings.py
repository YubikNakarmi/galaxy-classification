from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

from src.utils.yaml_utils import load_yaml

root = Path(__file__).resolve().parent.parent
load_dotenv(root / ".env")

class AzureStorageSettings(BaseModel):
    connection_string: str
    container_name: str ="image-uploads"  # default container name for image uploads


class DatabaseSettings(BaseModel):
    url: str
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class CelerySettings(BaseModel):
    broker_url: str
    result_backend: str


class Settings(BaseModel):
    database: DatabaseSettings
    azure_storage: AzureStorageSettings
    celery: CelerySettings
    environment: str = "development"
    debug: bool = False


@lru_cache 
def get_settings() -> Settings:
    raw = load_yaml(root / "configs" / "backend.yaml") #map yaml configs to Settings classes ( has to have same format)

    
    raw["database"]["url"] = os.getenv("DATABASE_URL","sqlite:///app.db") 
    raw["azure_storage"]["connection_string"] = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

    try:
        return Settings(**raw)#auto set kwargs
    except Exception as e:
        raise RuntimeError(f"Error occurred while creating Settings: {e}")


settings = get_settings()