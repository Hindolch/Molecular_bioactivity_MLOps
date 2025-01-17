import numpy as np
import logging
from src.model_evaluation import ModelEvaluator, XGBoostModelEvaluationStrategy
from prefect import task
import mlflow

@task
def model_evaluation_task(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Task function for evaluating a trained model using a specific strategy.

    Parameters:
    model: The trained model to evaluate.
    X_test (np.ndarray): The test dataset features.
    y_test (np.ndarray): The true labels for the test dataset.

    Returns:
    dict: Evaluation metrics.
    """
    try:
        logging.info("Starting model evaluation task")
        
        # Initialize the evaluation context and strategy
        evaluation_strategy = XGBoostModelEvaluationStrategy()
        evaluator = ModelEvaluator(strategy=evaluation_strategy)
        
        metrics = evaluator.evaluate(model=model, X_test=X_test, y_test=y_test)
        
        # Log metrics to MLflow
        with mlflow.start_run():
            logging.info("Logging metrics to MLflow")
            
            mlflow.log_metrics(metrics)  # Logs the metrics dictionary to MLflow
            logging.info("Metrics logged to MLflow")
        
        logging.info(f"Model evaluation completed. Metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise