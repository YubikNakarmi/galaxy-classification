from src.training import CheckpointManager
from pathlib import Path
from src.config import load_and_validate_config
from src.utils.yaml_utils import load_yaml
import datetime

root = Path(__file__).parent.parent
file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
cfg = load_and_validate_config(file_config, root=root )
experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  


local_checkpoint_dir = cfg.storage.local.checkpoints.dir
print(f"Local checkpoint directory: {local_checkpoint_dir}")

azure_dir = cfg.train.checkpoints.checkpoint_restore_path
print(f"Azure checkpoint directory: {azure_dir}")

local_restore = cfg.train.checkpoints.local_checkpoint_path
print(f"Local restore path: {local_restore}")