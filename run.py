import sys
from src.main import main
from src.model import ModelTrainer
from src.utils import logger, load_data
from config import DATASET_CONFIG, MODEL_CONFIG

if __name__ == "__main__":
    try:
        # Run the main pipeline (ETL -> EDA -> Model Training)
        main()
        
        # If a song name is provided as argument, generate recommendations
        if len(sys.argv) > 1:
            song_name = " ".join(sys.argv[1:])
            logger.info(f"Generating recommendations for song: {song_name}")
            
            # Load data and model
            model_trainer = ModelTrainer()
            df = model_trainer.load_data()
            model = model_trainer.load_model()
            
            if df is None or model is None:
                logger.error("Failed to load data or model for recommendations")
                sys.exit(1)
                
            # Get recommendations
            result = model_trainer.recommend_songs(song_name, df, model)
            
            if result is None:
                logger.error(f"No recommendations found for song: {song_name}")
                sys.exit(1)
                
            input_song, recommendations = result
            
            # Print results
            print(f"\nRecommendations for '{input_song['track_name'].values[0]}' by {input_song['artists'].values[0]}:")
            print(recommendations[['track_name', 'artists']].to_string(index=False))
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)