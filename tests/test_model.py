import pytest
from src.model import ModelTrainer
from src.utils import load_data
from config import MODEL_CONFIG
import os

class TestModelTrainer:
    @pytest.fixture
    def model_trainer(self):
        return ModelTrainer()
    
    def test_load_data(self, model_trainer):
        df = model_trainer.load_data()
        assert df is not None
        assert not df.empty
    
    def test_train_model(self, model_trainer):
        df = model_trainer.load_data()
        model = model_trainer.train_model(df)
        assert model is not None
    
    def test_save_load_model(self, model_trainer, tmp_path):
        # Set temporary model path
        test_file = tmp_path / "test_model.pkl"
        model_trainer.model_path = test_file
        
        df = model_trainer.load_data()
        model = model_trainer.train_model(df)
        success = model_trainer.save_model(model)
        
        assert success
        assert os.path.exists(test_file)
        
        loaded_model = model_trainer.load_model()
        assert loaded_model is not None
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
    
    def test_recommend_songs(self, model_trainer):
        df = model_trainer.load_data()
        model = model_trainer.train_model(df)
        
        # Test with a song that should exist
        result = model_trainer.recommend_songs("Shape of You", df, model)
        assert result is not None
        input_song, recommendations = result
        assert not input_song.empty
        assert not recommendations.empty
        assert len(recommendations) == MODEL_CONFIG['n_neighbors'] - 1
        
        # Test with a non-existent song
        result = model_trainer.recommend_songs("Nonexistent Song Name", df, model)
        assert result is None
    
    def test_full_pipeline(self, model_trainer, tmp_path):
        # Set temporary model path
        test_file = tmp_path / "test_model.pkl"
        model_trainer.model_path = test_file
        
        success = model_trainer.run()
        assert success
        assert os.path.exists(test_file)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)