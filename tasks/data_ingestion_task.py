from src.data_ingestion import ChemicalDataIngestion
from prefect import task
import pandas as pd

@task
def data_ingestion_task()->pd.DataFrame:
    data_ingestor = ChemicalDataIngestion()
    df = data_ingestor.initiate_data_ingestion()
    return df