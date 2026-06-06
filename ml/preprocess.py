
from src.dataset.dataset import GalaxyDataset
from src.dataset.label_preprocessing import Label_Processing_Classic
from sklearn.model_selection import train_test_split
from pathlib import Path
from dotenv import load_dotenv
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_ml_logging
import click



#setup logging and config load
root = Path(__file__).resolve().parent.parent
load_dotenv(root / '.env') 

log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)

file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)

preprocessed_dir = ml_yaml['data']['preprocessed_dir']
split_dir = ml_yaml['data']['split_dir']

active_variant = str(ml_yaml['active_variant'])
seed = ml_yaml["env"]["seed"]
test_size = ml_yaml['variants'][active_variant]['data']['split']["test_size"]
train_size = ml_yaml['variants'][active_variant]['data']['split']["train_size"]
val_size = ml_yaml['variants'][active_variant]['data']['split']["val_size"]
# label processing

#splitting data 




def split():

    
    
    X_t







def main():
    pass 

if __name__ == "__main__":
    main()