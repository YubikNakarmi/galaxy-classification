from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.utils.yaml_utils import load_yaml
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

root = Path(__file__).resolve().parent.parent

try:
    load_dotenv(root / ".env")
except Exception as e:
    print(f"Error loading .env file: {e}")


class CelerySettings(BaseModel):
    broker_url: str
    result_backend: str

class BentomlRoutes(BaseModel):
    default: str
    dirichlet: str
    hubble: str

class DatabaseSettings(BaseModel):
    url: str = Field(
        default_factory= lambda:os.getenv("DATABASE_URL", "sqlite:///app.db") #pydantic looks up the alias and uses env variable
    )
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class AzureStorageSettings(BaseModel):
    container_name: str
    connection_string: str = Field(
        default_factory= lambda:os.environ["AZURE_STORAGE_CONNECTION_STRING"] #pydantic looks up the alias and uses env variable
    )
class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        yaml_file =os.getenv("BACKEND_YAML_PATH", 
                             root / "configs" / "backend.yaml"),
        
        yaml_file_encoding = "utf-8",
        extra="allow",
        )
    
    database: DatabaseSettings
    azure_storage: AzureStorageSettings
    celery: CelerySettings
    environment: str = "development"
    debug: bool = False
    bentoml_routes: BentomlRoutes
    environment: str = "development"
    
    @classmethod
    def settings_customise_sources(  #override yaml configs if required
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,):
        
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings)



@lru_cache 
def get_settings() -> Settings:
   
    try:
        return Settings()#auto set kwargs
    except Exception as e:
        raise RuntimeError(f"Error occurred while creating Settings: {e}")


settings = get_settings()