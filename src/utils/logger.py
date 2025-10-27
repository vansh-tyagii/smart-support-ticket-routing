import os
import sys
import logging
import yaml

# logging_str = "[%(asctime)s] %(levelname)s - %(module)s - %(message)s"

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging_str = config['logging']['format']
log_dir = config['logging']['log_dir']
log_filename = config['logging']['log_filename']

full_log_path = os.path.join(log_dir, log_filename)
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("SSTR_Logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(full_log_path)
    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(logging_str)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)