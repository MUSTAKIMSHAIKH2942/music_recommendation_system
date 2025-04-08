import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import DATASET_CONFIG
from src.utils import logger, load_data, save_data, validate_data
from typing import Optional

class ETL:
    """Extract, Transform, Load (ETL) class for music data processing."""
    
    def __init__(self):
        self.raw_data_path = DATASET_CONFIG['raw_file']
        self.processed_data_path = DATASET_CONFIG['processed_file']
        self.features = DATASET_CONFIG['features']
        self.id_columns = DATASET_CONFIG['id_columns']
    
    def extract(self) -> Optional[pd.DataFrame]:
        """Extract data from the raw data file."""
        logger.info("Extracting raw data")
        df = load_data(self.raw_data_path)
        if df is None or not validate_data(df, self.id_columns + self.features):
            return None
        return df
    
    def transform(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transform the data by scaling features."""
        logger.info("Transforming data")
        try:
            # Keep original columns
            transformed_df = df[self.id_columns].copy()
            
            # Scale features
            scaler = StandardScaler()
            transformed_df[self.features] = scaler.fit_transform(df[self.features])
            
            return transformed_df
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            return None
    
    def load(self, df: pd.DataFrame) -> bool:
        """Load the processed data to a file."""
        logger.info("Loading processed data")
        if df is None:
            logger.error("No data to load")
            return False
        return save_data(df, self.processed_data_path)
    
    def run(self) -> bool:
        """Run the complete ETL pipeline."""
        logger.info("Starting ETL process")
        df = self.extract()
        if df is None:
            logger.error("ETL process failed at extraction stage")
            return False
            
        transformed_df = self.transform(df)
        if transformed_df is None:
            logger.error("ETL process failed at transformation stage")
            return False
            
        if not self.load(transformed_df):
            logger.error("ETL process failed at loading stage")
            return False
            
        logger.info("ETL process completed successfully")
        return True