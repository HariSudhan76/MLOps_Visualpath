import yaml
import json
from trainer.logger import logger

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved at {path}")
