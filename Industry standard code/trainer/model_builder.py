from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_REGISTRY = {
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression
}

def get_model(model_type, **kwargs):
    return MODEL_REGISTRY[model_type](**kwargs)
