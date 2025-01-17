from src.feature_engineering import MolecularFeatureEngineer, DFProperties
import pandas as pd
from prefect import task
import logging


@task
def feature_engineering_task(df: pd.DataFrame) -> pd.DataFrame:
    df_strategy = DFProperties()

    feature_engineer = MolecularFeatureEngineer(
        df_strategy=df_strategy    
        )

    try:
        processed_df = feature_engineer.execute_strategy(df)
        logging.info("Feature engineering task completed successfully")
        return processed_df
    except Exception as e:
        logging.error(f"Error in feature engineering task: {str(e)}")
        raise