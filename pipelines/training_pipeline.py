from tasks.data_ingestion_task import data_ingestion_task
from tasks.feature_engineering_task import feature_engineering_task
from tasks.data_splitting_task import data_splitter_task
from tasks.model_building_task import model_building_task
from tasks.model_evaluation_task import  model_evaluation_task
from src.model_building import XGBRegressionStrategy
import numpy as np
import logging
from prefect import flow

@flow
def chemical_pipeline():
    raw_data = data_ingestion_task()
    logging.info("Starting feature engineering")
    processed_data = feature_engineering_task(raw_data)
    processed_data = processed_data.dropna(subset=['pIC50'])
    logging.info(f"Feature engineering completed. Processed data shape: {processed_data.shape}")
    X_train, X_test, y_train, y_test = data_splitter_task(
        processed_data, target_column="pIC50"
    )

    # print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    # print(f"Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    # print(f"Number of NaN values in y_train: {np.isnan(y_train).sum()}")
    
    # print("FIRST 10 ROWS OF Y TRAINNNNNNN",y_train[:10])
    strategy = XGBRegressionStrategy()
    model = model_building_task(X_train, y_train,strategy=strategy)
    model_evaluation = model_evaluation_task(model, X_test, y_test)
    print("model trained")
    print(model_evaluation)

