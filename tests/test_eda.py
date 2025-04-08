import pytest
from src.eda import EDA
from src.utils import load_data
from config import VISUALIZATION_CONFIG
import os

class TestEDA:
    @pytest.fixture
    def eda(self):
        return EDA()
    
    def test_load_data(self, eda):
        df = eda.load_data()
        assert df is not None
        assert not df.empty
    
    def test_plot_feature_distributions(self, eda, tmp_path):
        # Set temporary visualization path
        test_file = tmp_path / "test_distributions.png"
        eda.viz_config['distributions_file'] = test_file
        
        df = eda.load_data()
        success = eda.plot_feature_distributions(df)
        
        assert success
        assert os.path.exists(test_file)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
    
    def test_plot_correlation_matrix(self, eda, tmp_path):
        # Set temporary visualization path
        test_file = tmp_path / "test_correlation.png"
        eda.viz_config['correlation_file'] = test_file
        
        df = eda.load_data()
        success = eda.plot_correlation_matrix(df)
        
        assert success
        assert os.path.exists(test_file)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
    
    def test_full_pipeline(self, eda, tmp_path):
        # Set temporary visualization paths
        eda.viz_config['distributions_file'] = tmp_path / "test_distributions.png"
        eda.viz_config['correlation_file'] = tmp_path / "test_correlation.png"
        eda.viz_config['pairplot_file'] = tmp_path / "test_pairplot.png"
        
        success = eda.run()
        assert success
        
        # Clean up
        for file in tmp_path.glob("*.png"):
            os.remove(file)