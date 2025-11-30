from trainer.data_loader import load_data
from trainer.trainer import train_model
from trainer.utils import load_config, save_metrics
from trainer.logger import logger

def main():
    config = load_config("config.yaml")
    df = load_data(config)
    metrics = train_model(df, config)
    save_metrics(metrics, config["output"]["metrics_path"])
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
