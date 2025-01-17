from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
import logging

# Abstract base class for model building strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Abstract method to build and train a model.

        Parameters:
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels/target.

        Returns:
        RegressorMixin: A trained model instance.
        """
        pass

# Concrete strategy for XGBRegressor using XGBoost
class XGBRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, np.ndarray):
            raise TypeError("X_train must be a NumPy array.")
        if not isinstance(y_train, np.ndarray):
            raise TypeError("y_train must be a NumPy array.")

        logging.info("Initializing XGBRegressor model")

        model = XGBRegressor(
            n_estimators=320,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
        )
        logging.info("Training XGBRegressor model.")
        model.fit(X_train, y_train)  # Fit the model to the training data

        logging.info("Model training completed.")
        return model
    
# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Parameters:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Executes the model building and training using the current strategy.

        Parameters:
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels/target.

        Returns:
        RegressorMixin: A trained model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)
