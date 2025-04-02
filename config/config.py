import os


class Config:
    """Base configuration class"""

    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-please-change-in-production")
    SQLALCHEMY_DATABASE_URI = "sqlite:////opt/sutazaiapp/database.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    SQLALCHEMY_ECHO = True


class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:////opt/sutazaiapp/test.db"


class ProductionConfig(Config):
    """Production configuration"""

    SECRET_KEY = os.environ.get("SECRET_KEY")

    # Database could be configured to use a different database in production
    # SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

    # Configure additional production settings
    SERVER_NAME = os.environ.get("SERVER_NAME")
    SSL_REDIRECT = os.environ.get("SSL_REDIRECT", False)


# Configuration dictionary to easily select configuration
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
