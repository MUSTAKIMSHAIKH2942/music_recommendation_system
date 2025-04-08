# Music Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A professional music recommendation system that suggests similar songs based on audio features using machine learning. The system implements the factory design pattern for modular and maintainable code.

## Features

- **ETL Pipeline**: Automated data extraction, transformation, and loading
- **Exploratory Data Analysis**: Automatic visualization of feature distributions and correlations
- **K-Nearest Neighbors Model**: Recommends similar songs based on audio features
- **Factory Pattern Implementation**: Clean separation of components
- **Comprehensive Logging**: Detailed execution logs for debugging
- **Automated Testing**: Unit tests for all components
- **Configuration Management**: Centralized settings for easy customization

## Project Structure

```
music_recommendation_system/
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   └── visualizations/     # Generated visualizations
├── models/                 # Trained model files
├── logs/                   # System logs
├── tests/                  # Unit tests
├── src/                    # Source code
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── run.py                  # Main execution script
└── README.md               # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-recommendation-system.git
   cd music-recommendation-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your Spotify tracks data in `data/raw/spotify_tracks.csv`

## Usage

### Run the complete pipeline (ETL → EDA → Model Training)
```bash
python run.py
```

### Get recommendations for a specific song
```bash
python run.py "Shape of You"
```

### Run unit tests
```bash
pytest tests/
```

### Configuration
Modify `config.py` to adjust:
- File paths
- Model parameters (number of neighbors, distance metric)
- Visualization settings
- Logging preferences

## Components

### 1. ETL (Extract, Transform, Load)
- Extracts raw Spotify track data
- Standardizes audio features using StandardScaler
- Saves processed data for analysis

### 2. EDA (Exploratory Data Analysis)
- Generates visualizations:
  - Feature distributions
  - Correlation matrix
  - Pair plots of selected features

### 3. Model Training
- Trains a K-Nearest Neighbors model
- Saves the trained model for recommendations
- Provides song recommendations based on audio features

## Technical Details

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Features Used**: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- **Distance Metric**: Euclidean distance
- **Default Neighbors**: 6 (5 recommendations)

## Example Output

```
Recommendations for 'Shape of You' by Ed Sheeran:

track_name                     artists
Perfect                        Ed Sheeran
Thinking Out Loud              Ed Sheeran
Photograph                     Ed Sheeran
Galway Girl                    Ed Sheeran
Happier                        Ed Sheeran
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

email - shaikhmustakim2942@gmail.com

Project Link: [https://github.com/yourusername/music-recommendation-system](https://github.com/yourusername/music-recommendation-system)

## Acknowledgments

- Spotify for providing the audio features dataset
- Scikit-learn for the machine learning algorithms
- Pandas for data manipulation
