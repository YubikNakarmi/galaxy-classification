from src.dataset.dataset import GalaxyDataset
from src.dataset.label_preprocessing import Label_Processing_Classic
from sklearn.model_selection import train_test_split
from pathlib import Path
from dotenv import load_dotenv
from src.utils import load_yaml, setup_ml_logging,yaml_helper


#setup logging and config load
root = Path(__file__).resolve().parent.parent.parent
load_dotenv(root / '.env') 
log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)

file_config = root / 'configs' / 'ml.yaml'
dir = yaml_helper(file_config)



print(help.parse_levels(levels=['data_dirs','train_file']))
# label processing

#splitting data 



def split():
    
