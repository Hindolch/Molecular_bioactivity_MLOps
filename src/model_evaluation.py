import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Abstract base class for model evaluation strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model, X_test: np.ndarray, y_test:np.ndarray)->dict:
        pass

# Concrete strategy for evaluating xgboost model
class XGBoostModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model, X_test:np.ndarray, y_test:np.ndarray):
        logging.info("Predicting using the trained model")
        model = joblib.load('Artifacts/model.pkl')
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics")
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        metrics = {"R2 Score": r2, "Mean Squared Error": mse, "Mean Absolute Error": mae}
        logging.info(f"Model Evaluation metrics: {metrics}")
        return metrics
    
# Context class for model evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        logging.info("Evaluating the model using the selected strategy")
        return self._strategy.evaluate_model(model, X_test, y_test)