from src.utils import load_yaml, setup_app_logging, setup_ml_logging
from .schema import *



def _parse_env(raw: dict) -> EnvConfig:
    e = raw["env"]
    return EnvConfig(
        name=e["name"],
        python_version=str(e["python_version"]),
        seed=int(e["seed"]),
        device=e["device"],
        num_workers=int(e["num_workers"]),
    )

