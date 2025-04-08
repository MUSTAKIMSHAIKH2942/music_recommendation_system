import pandas as pd
from sklearn.neighbors import NearestNeighbors
from config import DATASET_CONFIG, MODEL_CONFIG
from src.utils import logger, load_data, save_data, validate_data
from typing import Optional, Tuple

class ModelTrainer:
    """Model training and recommendation class."""
    
    def __init__(self):
        self.processed_data_path = DATASET_CONFIG['processed_file']
        self.model_path = MODEL_CONFIG['model_file']
        self.features = DATASET_CONFIG['features']
        self.id_columns = DATASET_CONFIG['id_columns']
        self.n_neighbors = MODEL_CONFIG['n_neighbors']
        self.metric = MODEL_CONFIG['metric']
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load the processed data."""
        logger.info("Loading data for model training")
        df = load_data(self.processed_data_path)
        if df is None or not validate_data(df, self.id_columns + self.features):
            return None
        return df
    
    def train_model(self, df: pd.DataFrame) -> Optional[NearestNeighbors]:
        """Train the KNN model."""
        logger.info("Training KNN model")
        try:
            knn = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric=self.metric
            )
            knn.fit(df[self.features])
            return knn
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def save_model(self, model: NearestNeighbors) -> bool:
        """Save the trained model."""
        logger.info("Saving trained model")
        return save_data(model, self.model_path)
    
    def load_model(self) -> Optional[NearestNeighbors]:
        """Load the trained model."""
        logger.info("Loading trained model")
        return load_data(self.model_path)
    
    # def recommend_songs(
    #     self, 
    #     song_name: str, 
    #     df: pd.DataFrame, 
    #     model: NearestNeighbors
    # ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    #     """
    #     Recommend songs similar to the given song.
        
    #     Args:
    #         song_name: Name of the song to find recommendations for
    #         df: DataFrame containing the song data
    #         model: Trained KNN model
            
    #     Returns:
    #         Tuple containing (input song, recommendations) or None if error occurs
    #     """
    #     logger.info(f"Finding recommendations for song: {song_name}")
    #     try:
    #         # Find the song (case insensitive)
    #         song = df[df['track_name'].str.lower() == song_name.lower()]
            
    #         if song.empty:
    #             logger.warning(f"Song not found: {song_name}")
    #             return None
                
    #         # Get recommendations
    #         # distances, indices = model.kneighbors(
    #         #     song[self.features].values.reshape(1, -1)
    #         # )
    #         input_features = song[self.features]
    #         distances, indices = model.kneighbors(input_features)

                        
    #         # Get recommended songs (excluding the input song itself)
    #         recommendations = df.iloc[indices[0][1:]]
            
    #         return song, recommendations
    #     except Exception as e:
    #         logger.error(f"Error generating recommendations: {str(e)}")
    #         return None
    def recommend_songs(
            self, 
            song_name: str, 
            df: pd.DataFrame, 
            model: NearestNeighbors
        ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
            logger.info(f"Finding recommendations for song: {song_name}")
            try:
                # Find the song (case insensitive)
                song = df[df['track_name'].str.lower() == song_name.lower()]
                
                if song.empty:
                    logger.warning(f"Song not found: {song_name}")
                    return None
                    
                # ✅ Only pass feature columns with names
                input_features = song[self.features]
                distances, indices = model.kneighbors(input_features)
                
                # Get recommended songs (excluding the input song itself)
                recommendations = df.iloc[indices[0][1:]]
                
                return song, recommendations
            except Exception as e:
                logger.error(f"Error generating recommendations: {str(e)}")
                return None

    
    def run(self) -> bool:
        """Run the complete model training pipeline."""
        logger.info("Starting model training process")
        df = self.load_data()
        if df is None:
            logger.error("Model training failed at data loading stage")
            return False
            
        model = self.train_model(df)
        if model is None:
            logger.error("Model training failed at model training stage")
            return False
            
        if not self.save_model(model):
            logger.error("Model training failed at model saving stage")
            return False
            
        # Test the model with a sample song
        test_song = "Shape of You"
        result = self.recommend_songs(test_song, df, model)
        
        if result is None:
            logger.warning(f"Failed to generate recommendations for test song: {test_song}")
        else:
            input_song, recommendations = result
            logger.info(f"Recommendations for '{test_song}':")
            logger.info("\n" + recommendations[self.id_columns].to_string(index=False))
            
        logger.info("Model training process completed successfully")
        return True  
