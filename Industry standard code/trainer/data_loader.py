import pandas as pd
from trainer.logger import logger
from trainer.utils import load_config

def load_data(config):
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(config["data"]["input_csv"])
        logger.info(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error while loading dataset: {e}")
        raise e
