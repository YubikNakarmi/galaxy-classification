
import yaml
from pathlib import Path


def load_yaml(path: str | Path):
    """
    Load a YAML file and return its contents as a Python dict.

    Parameters:
        path (str | Path): Path to YAML file

    Returns:
        dict: Parsed YAML content
    """
    
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {path}: {e}")