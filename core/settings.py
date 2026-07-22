from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

from src.utils.yaml_utils import load_yaml

root = Path(__file__).resolve().parent.parent
load_dotenv(root / ".env")


class DatabaseSettings(BaseModel):
    url: str
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class Settings(BaseModel):
    database: DatabaseSettings

@lru_cache
def get_settings() -> Settings:
    raw = load_yaml(root / "configs" / "backend.yaml")

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    raw["storage"]["database"]["url"] = db_url
    return Settings(**raw)


settings = get_settings()