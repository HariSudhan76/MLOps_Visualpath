from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL_REGISTRY = {
    "RandomForest": RandomForestClassifier,
    "DecisionTree": DecisionTreeClassifier
}

def get_model(model_type, **kwargs):
    return MODEL_REGISTRY[model_type](**kwargs)
