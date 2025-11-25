from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import os
import json
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from trainer.logger import logger
from trainer.model_builder import get_model
from trainer.preprocessing import build_preprocessor
from trainer.evaluator import evaluate_classification


def train_model(df, config):

    # ---------- Common preprocessing for all models ----------
    X = df.drop(config["data"]["target"], axis=1)
    y = df[config["data"]["target"]]

    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    preprocessor = build_preprocessor(
        X_train, scaling=config["training"]["scaling"]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    os.makedirs("artifacts", exist_ok=True)

    # ---------- LOOP OVER ALL MODELS ----------
    for model_type in config["models"]:
        print(f"\n Training Model: {model_type}\n")

        # Set MLflow experiment (each model gets its OWN experiment)
        mlflow.set_experiment(f"{model_type}_Experiment")

        # Get model constructor
        model = get_model(model_type)

        # Get correct hyperparameters
        param_grid = config["hyperparameters"][model_type]

        # Get correct registry name
        registry_name = config["mlflow"]["model_registry_map"][model_type]

        with mlflow.start_run():

            mlflow.log_param("model_type", model_type)

            # Save hyperparameters used
            with open(f"artifacts/{model_type}_params.json", "w") as f:
                json.dump(param_grid, f)
            mlflow.log_artifact(f"artifacts/{model_type}_params.json")

            logger.info(f"Running GridSearchCV for {model_type}...")

            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring="f1_macro",
                n_jobs=-1
            )

            grid.fit(X_train_transformed, y_train)
            best_model = grid.best_estimator_

            # Log best hyperparameters
            mlflow.log_params(grid.best_params_)

            # Evaluate
            metrics = evaluate_classification(best_model, X_test_transformed, y_test)
            mlflow.log_metrics({
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"]
            })

            # Save confusion matrix
            with open(f"artifacts/{model_type}_confusion.json", "w") as f:
                json.dump(metrics["confusion_matrix"], f)
            mlflow.log_artifact(f"artifacts/{model_type}_confusion.json")

            # Save model file
            model_path = f"artifacts/{model_type}_model.pkl"
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(model_path)

            # Signature + input_example required for Inputs/Outputs tab
            signature = infer_signature(X_train_transformed, y_train)

            # REGISTER MODEL in MLflow Registry
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=registry_name,
                signature=signature,
                input_example=X_train_transformed[:5]
            )

            logger.info(f"✔ Registered model under registry name: {registry_name}")


    logger.info(" All models trained and logged successfully!")
