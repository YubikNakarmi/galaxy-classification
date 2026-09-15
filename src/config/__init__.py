from __future__ import annotations
 
from pathlib import Path
 
from src.utils.yaml_utils import load_yaml
from src.config.schema import RootConfig
from .parser import parse_config
 

__all__ = ["load_and_validate_config", "RootConfig"]


def load_and_validate_config(
    ml_yaml_path: Path | str,
    root: Path | str | None = None,
    env_path: Path | str | None = None,
) -> RootConfig:
    """
    Load, validate, and parse the project config in one call.

    Args:
        ml_yaml_path: path to configs/ml.yaml
        root: project root for resolving relative paths.
              Defaults to two levels above ml_yaml_path (configs/../..)
        env_path: path to .env file. Defaults to root/.env

    Returns:
        RootConfig object

    Raises:
        ConfigError: if any validation check fails
    """
    ml_yaml_path = Path(ml_yaml_path)
    root = Path(root) if root else ml_yaml_path.parent.parent
    env_path = Path(env_path) if env_path else None

    raw= load_yaml(ml_yaml_path)
    #validate_config(raw, root)

    return parse_config(raw, root)