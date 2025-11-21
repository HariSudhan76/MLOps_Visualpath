from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
from trainer.logger import logger
from trainer.model_builder import get_model
from trainer.evaluator import evaluate_classification
from trainer.preprocessing import build_preprocessor


def train_model(df, config):

    try:
        logger.info("Splitting dataset...")
        
        X = df.drop(config["data"]["target"], axis=1)
        y = df[config["data"]["target"]]



        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
            stratify=y
        )
        
        preprocessor = build_preprocessor(X_train, scaling=config["training"]["scaling"])
        # scaler = None
        # if config["training"]["scaling"]:
        #     logger.info("Applying StandardScaler...")
        #     scaler = StandardScaler()
        #     X_train = scaler.fit_transform(X_train)
        #     X_test = scaler.transform(X_test)

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        logger.info("Starting hyperparameter tuning...")

        model = get_model(config["model"]["type"])

        param_grid = config["model"]["hyperparameters"]
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )

        grid_search.fit(X_train_transformed, y_train)

        best_model = grid_search.best_estimator_

        logger.info(f"Best model: {best_model}")
        logger.info(f"Best params: {grid_search.best_params_}")

        metrics = evaluate_classification(best_model, X_test_transformed, y_test)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(best_model, config["output"]["model_path"])
        # if scaler:
        #     joblib.dump(scaler, config["output"]["scaler_path"])

        return metrics

    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise e
