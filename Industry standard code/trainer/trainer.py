from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import os
import json
import mlflow
import mlflow.sklearn

from trainer.logger import logger
from trainer.model_builder import get_model
from trainer.evaluator import evaluate_classification
from trainer.preprocessing import build_preprocessor


def train_model(df, config):
    try:
        logger.info("Starting training pipeline...")

        # Set MLflow Experiment
        model_type = config["model"]["type"]
        mlflow.set_experiment(f"{model_type}_Experiment")

        # Split features and target
        X = df.drop(config["data"]["target"], axis=1)
        y = df[config["data"]["target"]]

        logger.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
            stratify=y
        )

        # Build preprocessing
        preprocessor = build_preprocessor(
            X_train, scaling=config["training"]["scaling"]
        )

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        model = get_model(config["model"]["type"])
        param_grid = config["model"]["hyperparameters"]

        # CREATE ARTIFACT DIR
        os.makedirs("artifacts", exist_ok=True)

        # START MLFLOW RUN
        with mlflow.start_run():

            mlflow.log_param("model_type", config["model"]["type"])

            # Log hyperparameter search space as artifact
            with open("artifacts/hyperparameter_space.json", "w") as f:
                json.dump(param_grid, f)
            mlflow.log_artifact("artifacts/hyperparameter_space.json")

            logger.info("Running GridSearchCV...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=3,
                n_jobs=-1
            )
            grid_search.fit(X_train_transformed, y_train)

            best_model = grid_search.best_estimator_

            # Log best hyperparameters
            mlflow.log_params(grid_search.best_params_)

            # Evaluate model
            logger.info("Evaluating model...")
            metrics = evaluate_classification(best_model, X_test_transformed, y_test)

            # Log numeric metrics
            mlflow.log_metrics({
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"]
            })

            # Save confusion matrix as artifact
            with open("artifacts/confusion_matrix.json", "w") as f:
                json.dump(metrics["confusion_matrix"], f)
            mlflow.log_artifact("artifacts/confusion_matrix.json")

            # Save model
            joblib.dump(best_model, config["output"]["model_path"])
            mlflow.log_artifact(config["output"]["model_path"])

            # Save preprocessor
            joblib.dump(preprocessor, config["output"]["preprocessor_path"])
            mlflow.log_artifact(config["output"]["preprocessor_path"])

            # ⭐ REGISTER MODEL IN MLFLOW MODEL REGISTRY
            # mlflow.sklearn.log_model(
            #     sk_model=best_model,
            #     name="model",
            #     registered_model_name=config["mlflow"]["model_name"]
            # )
            model_registry_name = config["mlflow"]["model_registry_map"][model_type]

            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train_transformed, y_train)

            mlflow.sklearn.log_model(
                sk_model = best_model,
                artifact_path="model",
                registered_model_name=model_registry_name,
                signature=signature,
                input_example=X_train_transformed[:5]

            )
            print(f"model registered: {model_registry_name}")

        logger.info("Training completed successfully.")
        return metrics

    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise e
