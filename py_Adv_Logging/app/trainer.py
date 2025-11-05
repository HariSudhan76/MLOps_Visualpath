import time

class Trainer:
    def __init__(self, logger):
        self.logger = logger

    def train(self):
        self.logger.info("Training pipeline initiated...")
        try:
            for epoch in range(1, 4):
                self.logger.debug(f"Epoch {epoch}: processing data...")
                time.sleep(1)
            self.logger.info("Training completed successfully!")
        except Exception as e:
            self.logger.exception(f"Training failed: {e}")
