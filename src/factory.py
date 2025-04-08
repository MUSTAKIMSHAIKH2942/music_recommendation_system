from src.etl import ETL
from src.eda import EDA
from src.model import ModelTrainer
from src.utils import logger

class ComponentFactory:
    """Factory class for creating components of the music recommendation system."""
    
    @staticmethod
    def get_component(component_type: str):
        """
        Get a component instance based on the component type.
        
        Args:
            component_type: Type of component to create ('etl', 'eda', 'model')
            
        Returns:
            An instance of the requested component
            
        Raises:
            ValueError: If an invalid component type is provided
        """
        components = {
            'etl': ETL,
            'eda': EDA,
            'model': ModelTrainer
        }
        
        try:
            logger.info(f"Creating component: {component_type}")
            return components[component_type.lower()]()
        except KeyError:
            error_msg = f"Invalid component type: {component_type}. Must be one of {list(components.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)