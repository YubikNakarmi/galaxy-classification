
import yaml
from pathlib import Path


def load_yaml(path: str | Path)-> dict:
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
        

class yaml_helper:

    def __init__(self,file_loc) -> None:
        self.loc = file_loc

    def load(self):
        """
    Load a YAML file and return its contents as a Python dict.

    Parameters:
        path (str | Path): Path to YAML file

    Returns:
        dict: Parsed YAML content
    """
    
        path = Path(self.loc)

        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format in {path}: {e}")
            
    def parse_levels(self,levels:list[str]):

        '''
        Parse a YAML file and return a nested dictionary based on specified levels.

        Parameters:
            levels (list[str]): List of keys representing the hierarchy to navigate in the YAML structure.
        '''

        data = self.load()

        for level in levels:
            if level in data:
                data = data[level]
            else:
                raise KeyError(f"Level '{level}' not found in YAML structure.")
        return data
    

root = Path(__file__).resolve().parent.parent

file = root / 'configs' / 'ml.yaml'


