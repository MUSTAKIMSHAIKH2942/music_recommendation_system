# import os
# from pathlib import Path

# # Base directory
# BASE_DIR = Path(__file__).parent.parent

# # Data paths
# DATA_DIR = BASE_DIR / 'data'
# RAW_DATA_DIR = DATA_DIR / 'raw'
# PROCESSED_DATA_DIR = DATA_DIR / 'processed'
# VISUALIZATIONS_DIR = DATA_DIR / 'visualizations'

# # Model paths
# MODELS_DIR = BASE_DIR / 'models'

# # Log paths
# LOGS_DIR = BASE_DIR / 'logs'

# # Create directories if they don't exist
# for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VISUALIZATIONS_DIR, MODELS_DIR, LOGS_DIR]:
#     directory.mkdir(parents=True, exist_ok=True)

# # Dataset configuration
# DATASET_CONFIG = {
#     'raw_file': RAW_DATA_DIR / 'spotify_tracks.csv',
#     'processed_file': PROCESSED_DATA_DIR / 'processed_spotify_tracks.csv',
#     'features': [
#         'danceability', 'energy', 'key', 'loudness', 'mode', 
#         'speechiness', 'acousticness', 'instrumentalness', 
#         'liveness', 'valence', 'tempo'
#     ],
#     'id_columns': ['track_id', 'track_name', 'artists'],
#     'target': None  # No target for unsupervised learning
# }

# # Model configuration
# MODEL_CONFIG = {
#     'model_file': MODELS_DIR / 'knn_model.pkl',
#     'n_neighbors': 6,
#     'metric': 'euclidean'
# }

# # Visualization configuration
# VISUALIZATION_CONFIG = {
#     'distributions_file': VISUALIZATIONS_DIR / 'feature_distributions.png',
#     'correlation_file': VISUALIZATIONS_DIR / 'correlation_matrix.png',
#     'pairplot_file': VISUALIZATIONS_DIR / 'pairplot.png',
#     'features_to_plot': ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness']
# }

# # Logging configuration
# LOGGING_CONFIG = {
#     'log_file': LOGS_DIR / 'music_recommendation.log',
#     'log_level': 'INFO',
#     'max_bytes': 1048576,  # 1MB
#     'backup_count': 5
# }


import os
from pathlib import Path

# Dynamically locate project root (assumes this file is in src/)
PROJECT_NAME = "music_recommendation_system"
BASE_DIR = Path(__file__).resolve()

while BASE_DIR.name != PROJECT_NAME and BASE_DIR != BASE_DIR.parent:
    BASE_DIR = BASE_DIR.parent

if BASE_DIR.name != PROJECT_NAME:
    raise RuntimeError(f"Could not locate project root named '{PROJECT_NAME}'.")

# === Directory Structure ===
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
VISUALIZATIONS_DIR = DATA_DIR / 'visualizations'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# === Ensure Required Directories Exist ===
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VISUALIZATIONS_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# === Dataset Configuration ===
DATASET_CONFIG = {
    'raw_file': RAW_DATA_DIR / 'spotify_tracks.csv',
    'processed_file': PROCESSED_DATA_DIR / 'processed_spotify_tracks.csv',
    'features': [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ],
    'id_columns': ['track_id', 'track_name', 'artists'],
    'target': None  # No target for unsupervised learning
}

# === Model Configuration ===
MODEL_CONFIG = {
    'model_file': MODELS_DIR / 'knn_model.pkl',
    'n_neighbors': 6,
    'metric': 'euclidean'
}

# === Visualization Configuration ===
VISUALIZATION_CONFIG = {
    'distributions_file': VISUALIZATIONS_DIR / 'feature_distributions.png',
    'correlation_file': VISUALIZATIONS_DIR / 'correlation_matrix.png',
    'pairplot_file': VISUALIZATIONS_DIR / 'pairplot.png',
    'features_to_plot': [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness'
    ]
}

# === Logging Configuration ===
LOGGING_CONFIG = {
    'log_file': LOGS_DIR / 'music_recommendation.log',
    'log_level': 'INFO',
    'max_bytes': 1_048_576,  # 1MB
    'backup_count': 5
}
