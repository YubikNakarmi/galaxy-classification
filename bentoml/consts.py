from torchvision import transforms
import random
from pathlib import Path
from src.utils.yaml_utils import load_yaml
import torch

root = Path(__file__).parent.parent
file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)


 
DEFAULT_CLASS_NAMES = {
    0: "completely_round_smooth",
    1: "in_between_smooth",
    2: "cigar_shaped_smooth",
    3: "edge_on",
    4: "spiral",
}


print(DEFAULT_CLASS_NAMES[0])
 
# HUBBLE_CLASSES = [
#     "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7",
#     "S0",
#     "Sa", "Sb", "Sc", "Sd",
#     "SBa", "SBb", "SBc", "SBd",
#     "Irr",
# ]
 
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1376, 0.1276, 0.1187],
        std=[0.1728, 0.1586, 0.1488],
    ),
])




DEVICE = ml_yaml["env"]["device"]
experiment_root_default = ml_yaml["variants"]["Default"]["experiment"]["root"] #MLFLOW experiment name for a specific variant
experiment_root_hubble = ml_yaml["variants"]["Hubble"]["experiment"]["root"]
experiment_root_dirichlet = ml_yaml["variants"]["Dirichlet"]["experiment"]["root"]
device = ml_yaml["env"]["device"]

