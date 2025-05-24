import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-for-portfolio-site')
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    DEVELOPMENT = True

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True

class ProductionConfig(Config):
    """Production configuration."""
    # Ensure a strong secret key is set in production
    SECRET_KEY = os.environ.get('SECRET_KEY')

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# Get configuration based on environment variable
def get_config():
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])