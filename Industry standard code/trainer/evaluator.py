from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from trainer.logger import logger
import json

def evaluate_classification(model, X_test, y_test):
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    logger.info(f"Metrics: {metrics}")
    return metrics
