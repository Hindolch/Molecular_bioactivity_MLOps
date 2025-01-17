import os, sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict
import psycopg2
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

@dataclass
class DataIngestionConfig:
    # get the src directory path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # get the project parent directory
    project_root = os.path.dirname(src_dir)
    data_artifacts_dir = os.path.join(project_root, "Data")
    chem_data_path: str = os.path.join(data_artifacts_dir, "chem_dataset.csv")


class ChemicalDataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

        #database connectivity parameters
        self.db_params = {
            'database': 'mlchem',
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'admin@123'),
            'host': 'localhost',
            'port': '5432'
        }

    def get_database_connection(self):
        """Create and return database connection engine"""
        try:
            encoded_password = "admin%40123" 
            connection_string = f"postgresql+psycopg2://{self.db_params['user']}:{encoded_password}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"
            return create_engine(connection_string)
        except Exception as e:
            logging.error(f"Error creating database connection {str(e)}")
            raise e
        
    def get_data_from_table(self, table_name:str, engine)->pd.DataFrame:
        """Fetch data from a specified table"""
        try:
            query = f"SELECT * FROM public.{table_name}"
            logging.info(f"Fetching data from table {table_name}")
            df = pd.read_sql(query, engine)

            if df.empty:
                raise ValueError(f"No data retrieved from table {table_name}")

            logging.info(f"Successfully retrieved data from {table_name}")
            return df
        
        except Exception as e:
            logging.error(f"Error fetching data from {table_name}")
            raise e
        
    
    def save_dataframe(self, df:pd.DataFrame, file_path:str):
        """Save dataframe to CSV file"""
        try:
            df.to_csv(file_path, index=False)
            logging.info(f"Data successfully saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving file to the destination{file_path}")
            raise e
    
    def initiate_data_ingestion(self)->Tuple[str]:
        """Initiate data ingestion process for both tables"""
        try:
            # create artifacts directory
            os.makedirs(self.ingestion_config.data_artifacts_dir, exist_ok= True)

            # Get database connection
            engine = self.get_database_connection()

            try:
                #fetch data from table
                chem_df = self.get_data_from_table('chem_db3', engine)

                #save both datasets
                if not chem_df.empty:
                    self.save_dataframe(df=chem_df, file_path=self.ingestion_config.chem_data_path)
                    logging.info(f"Chemical dataset preview:\n{chem_df.head()}")

                    return chem_df            
            finally:
                engine.dispose()
        except Exception as e:
            logging.error(f"Error in data ingestion {str(e)}")
            raise e
        
# # Example usage
# if __name__ == "__main__":
#     ingestion = ChemicalDataIngestion()
#     chem_path = ingestion.initiate_data_ingestion()
#     print("Chemical data path:", chem_path)

