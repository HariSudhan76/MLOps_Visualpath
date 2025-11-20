import logging
import os

def setup_logger(log_file="training.log"):
    logger = logging.getLogger("MLOpsPipeline")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(f"logs/{log_file}")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()
