import logging
import logging.config
from pathlib import Path
import yaml
import pandas as pd
import pickle
from typing import Any, Dict, Optional
from config import LOGGING_CONFIG

def setup_logging():
    """Set up logging configuration."""
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'file_handler': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': LOGGING_CONFIG['log_level'],
                'formatter': 'standard',
                'filename': LOGGING_CONFIG['log_file'],
                'maxBytes': LOGGING_CONFIG['max_bytes'],
                'backupCount': LOGGING_CONFIG['backup_count'],
                'encoding': 'utf8'
            },
            'console_handler': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            '': {
                'handlers': ['file_handler', 'console_handler'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    })
    return logging.getLogger(__name__)

logger = setup_logging()

def load_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load data from a file."""
    try:
        logger.info(f"Loading data from {file_path}")
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return None
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None

def save_data(data: Any, file_path: Path) -> bool:
    """Save data to a file."""
    try:
        logger.info(f"Saving data to {file_path}")
        if file_path.suffix == '.csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
                return True
        elif file_path.suffix == '.pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                return True
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return False
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        return False

def validate_data(df: pd.DataFrame, expected_columns: list) -> bool:
    """Validate that the DataFrame contains the expected columns."""
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing columns in DataFrame: {missing_columns}")
        return False
    return True