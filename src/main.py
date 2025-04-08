from src.factory import ComponentFactory
from src.utils import logger

def main(components: list = None):
    """
    Main function to run the music recommendation system.
    
    Args:
        components: List of components to run ('etl', 'eda', 'model'). 
                   If None, runs all components.
    """
    logger.info("Starting music recommendation system")
    
    # Default to running all components if none specified
    if components is None:
        components = ['etl', 'eda', 'model']
    
    # Run each component
    for component in components:
        try:
            logger.info(f"Processing component: {component}")
            obj = ComponentFactory.get_component(component)
            success = obj.run()
            
            if not success:
                logger.warning(f"Component {component} completed with errors")
            else:
                logger.info(f"Component {component} completed successfully")
                
        except Exception as e:
            logger.error(f"Error processing component {component}: {str(e)}")
            continue
    
    logger.info("Music recommendation system completed")
