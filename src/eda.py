import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATASET_CONFIG, VISUALIZATION_CONFIG
from src.utils import logger, load_data, validate_data
from typing import Optional

class EDA:
    """Exploratory Data Analysis (EDA) class for music data."""
    
    def __init__(self):
        self.processed_data_path = DATASET_CONFIG['processed_file']
        self.features = DATASET_CONFIG['features']
        self.id_columns = DATASET_CONFIG['id_columns']
        self.viz_config = VISUALIZATION_CONFIG
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load the processed data."""
        logger.info("Loading data for EDA")
        df = load_data(self.processed_data_path)
        if df is None or not validate_data(df, self.id_columns + self.features):
            return None
        return df
    
    def plot_feature_distributions(self, df: pd.DataFrame) -> bool:
        """Plot distributions of selected features."""
        logger.info("Plotting feature distributions")
        try:
            plt.figure(figsize=(12, 8))
            df[self.viz_config['features_to_plot']].hist(bins=20, layout=(2, 3), figsize=(15, 10))
            plt.tight_layout()
            plt.savefig(self.viz_config['distributions_file'])
            plt.close()
            logger.info(f"Saved feature distributions to {self.viz_config['distributions_file']}")
            return True
        except Exception as e:
            logger.error(f"Error plotting feature distributions: {str(e)}")
            return False
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> bool:
        """Plot correlation matrix of features."""
        logger.info("Plotting correlation matrix")
        try:
            plt.figure(figsize=(12, 10))
            corr = df[self.features].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.viz_config['correlation_file'])
            plt.close()
            logger.info(f"Saved correlation matrix to {self.viz_config['correlation_file']}")
            return True
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
            return False
    
    def plot_pairplot(self, df: pd.DataFrame) -> bool:
        """Plot pairplot of selected features."""
        logger.info("Plotting pairplot")
        try:
            sns.pairplot(df[self.viz_config['features_to_plot']])
            plt.savefig(self.viz_config['pairplot_file'])
            plt.close()
            logger.info(f"Saved pairplot to {self.viz_config['pairplot_file']}")
            return True
        except Exception as e:
            logger.error(f"Error plotting pairplot: {str(e)}")
            return False
    
    def run(self) -> bool:
        """Run the complete EDA pipeline."""
        logger.info("Starting EDA process")
        df = self.load_data()
        if df is None:
            logger.error("EDA process failed at data loading stage")
            return False
        
        success = True
        if not self.plot_feature_distributions(df):
            success = False
        if not self.plot_correlation_matrix(df):
            success = False
        if not self.plot_pairplot(df):
            success = False
            
        if success:
            logger.info("EDA process completed successfully")
        else:
            logger.warning("EDA process completed with some errors")
            
        return success  
