"""
Configuration management for Walmart Sales Forecasting Dashboard
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://walmart_user:walmart_password@localhost:5432/walmart_forecasting"
    DATABASE_TEST_URL: Optional[str] = None
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "walmart_forecasting"
    MODEL_ARTIFACT_PATH: str = "./mlflow/artifacts"
    
    # External APIs
    WEATHER_API_KEY: Optional[str] = None
    ECONOMIC_API_KEY: Optional[str] = None
    
    # Model Configuration
    FORECAST_HORIZON: int = 12
    VALIDATION_WINDOW: int = 52
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Override with environment variables if present
if os.getenv("DATABASE_URL"):
    settings.DATABASE_URL = os.getenv("DATABASE_URL")
if os.getenv("MLFLOW_TRACKING_URI"):
    settings.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if os.getenv("MLFLOW_EXPERIMENT_NAME"):
    settings.MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

