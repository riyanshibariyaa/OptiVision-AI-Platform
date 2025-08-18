import os
import logging

logger = logging.getLogger(__name__)

class Config:
    # Base directory - one level up from config.py
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    LOCAL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_images')
    # Make sure the folder exists
    os.makedirs(LOCAL_FOLDER, exist_ok=True)
    
    # Storage paths
    STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
    UPLOAD_FOLDER = os.path.join(STORAGE_DIR, 'uploads')
    PROCESSED_FOLDER = os.path.join(STORAGE_DIR, 'processed')
    MODEL_FOLDER = os.path.join(STORAGE_DIR, 'models')
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Model settings
    MODEL_CONFIDENCE_THRESHOLD = 0.4
    
    # Processing settings
    MAX_KERNEL_SIZE = 51  # Maximum Gaussian blur kernel size
    MIN_KERNEL_SIZE = 15  # Minimum Gaussian blur kernel size
    
    # Create necessary directories
    @classmethod
    def init_directories(cls):
        """Initialize required directories for the application"""
        try:
            os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
            os.makedirs(cls.PROCESSED_FOLDER, exist_ok=True)
            os.makedirs(cls.MODEL_FOLDER, exist_ok=True)
            logger.info("Application directories initialized")
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise