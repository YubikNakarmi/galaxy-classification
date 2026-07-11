from pathlib import Path
from src.utils.yaml_utils import load_yaml
from src.config import load_and_validate_config
root = Path(__file__).resolve().parent.parent

file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
cfg = load_and_validate_config(file_config)

print(cfg.variant.experiment.root)
