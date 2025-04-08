import pytest
from src.etl import ETL
from src.utils import load_data
from config import DATASET_CONFIG, PROCESSED_DATA_DIR
import pandas as pd
import os

class TestETL:
    @pytest.fixture
    def etl(self):
        return ETL()
    
    def test_extract(self, etl):
        df = etl.extract()
        assert df is not None
        assert not df.empty
        assert all(col in df.columns for col in DATASET_CONFIG['id_columns'] + DATASET_CONFIG['features'])
    
    def test_transform(self, etl):
        df = etl.extract()
        transformed_df = etl.transform(df)
        assert transformed_df is not None
        assert not transformed_df.empty
        assert all(col in transformed_df.columns for col in DATASET_CONFIG['id_columns'] + DATASET_CONFIG['features'])
    
    def test_load(self, etl, tmp_path):
        # Create a temporary processed file path
        test_file = tmp_path / "test_processed.csv"
        etl.processed_data_path = test_file
        
        df = etl.extract()
        transformed_df = etl.transform(df)
        success = etl.load(transformed_df)
        
        assert success
        assert os.path.exists(test_file)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
    
    def test_full_pipeline(self, etl, tmp_path):
        # Create a temporary processed file path
        test_file = tmp_path / "test_processed.csv"
        etl.processed_data_path = test_file
        
        success = etl.run()
        assert success
        assert os.path.exists(test_file)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)