from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import RootConfig


ROOT = Path(__file__).resolve().parent.parent.parent 

@dataclass
class Context:
    '''Context class to hold shared information for CLI commands.'''
    root: Path
    cfg: RootConfig
    mlflow_tracking_uri: str
    image_dir: Path

def load_context( *,
    service: str = "ml",
    env_path: Path | None = None,
    ml_yaml_path: Path | None = None,
    core_yaml_path: Path | None = None,)-> Context:

    ''' Core context file used my ML CLI to avoid reloading config and logging setup in each command. '''
    from dotenv import load_dotenv
    from loguru import logger

    from src.utils.yaml_utils import load_yaml
    from src.utils.logging_config import setup_ml_logging, setup_app_logging
    from src.config import load_and_validate_config

    env_path = env_path or ROOT / ".env"
    ml_yaml_path = ml_yaml_path or ROOT / "configs" / "ml.yaml"
    core_yaml_path = core_yaml_path or ROOT / "configs" / "core.yaml"

    load_dotenv(env_path)

    core_cfg = load_yaml(core_yaml_path)
    image_dir = Path(core_cfg["dirs"]["data"]["images"])
    if service == "ml":
        setup_ml_logging(core_cfg)
    else:
        json_logs = os.getenv("LOG_JSON", "false").lower() in ("true", "1", "yes")
        setup_app_logging(core_cfg, service=service, json_logs=json_logs)

    cfg = load_and_validate_config(ml_yaml_path, root=ROOT, env_path=env_path)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    logger.bind(component="bootstrap")
    

    return Context(root=ROOT, cfg=cfg, 
                   mlflow_tracking_uri=mlflow_uri, image_dir=image_dir)