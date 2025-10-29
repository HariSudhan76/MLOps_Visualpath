import yaml
import argparse

# 1. Load default values from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Define CLI overrides
parser = argparse.ArgumentParser(description="ML pipeline launcher")

parser.add_argument("--epochs", type=int, default=config["training"]["epochs"])
parser.add_argument("--lr", type=float, default=config["training"]["learning_rate"])
parser.add_argument("--model", type=str,choices=config["model"]["type"], default=config["model"]["type"][0], \
                       help = 'Choose model type from the list - ["RandomForest","Linear Regression","SVM"]' )


args = parser.parse_args()

print("Training model:", args.model)
print("Epochs:", args.epochs)
print("Learning Rate:", args.lr)
