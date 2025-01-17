from prefect import task
import numpy as np
from src.model_building import ModelBuildingStrategy, XGBRegressionStrategy
from sklearn.base import RegressorMixin
import mlflow, os, joblib, logging
from xgboost import XGBRegressor

@task
def model_building_task(X_train:np.ndarray, y_train:np.ndarray, strategy: ModelBuildingStrategy):
    """
    Prefect task to train a model and log it using MLflow.

    Parameters:
    - X_train (pd.DataFrame): Training data features.
    - y_train (pd.Series): Training labels.
    - strategy (ModelBuildingStrategy): Model training strategy.

    Returns:
    - dict: Contains model path and model object.
    """

    model = XGBRegressor(n_estimators=320,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,)
    
    experiment_name = "Chemical model training"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        try:
            mlflow.xgboost.autolog()
            model.fit(X_train,y_train)
            os.makedirs("Artifacts", exist_ok=True)
            model_path = "Artifacts/model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            logging.info(f"Model saved: {model_path}")
            return {"model_path": model_path, "model_object": model}
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise e
        finally:
            mlflow.end_run()