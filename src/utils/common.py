from box import ConfigBox 
from pathlib import Path 
from src.utils.logger import logger
from box.exceptions import BoxValueError 
import os
import yaml

"""" used for reading yaml files and creating directories """

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully.")
            return ConfigBox(content)  
        
    except BoxValueError:
        raise ValueError("YAML file is empty")    
    except Exception as e:
        raise e


def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")