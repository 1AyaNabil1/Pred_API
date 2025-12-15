"""
Configuration management for the MLflow Model Registry Proxy.
Handles environment variables and application settings.
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_TITLE: str = "MLflow Model Registry Proxy"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Security
    API_KEY: str = ""  # Leave empty to disable API key authentication
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]  # Frontend URLs
    
    # MLflow Configuration
    MLFLOW_BASE_URL: str = "http://localhost:5001"
    MLFLOW_AUTH_TOKEN: str = ""
    
    # File Upload Limits
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_EXTENSIONS: List[str] = [".pkl", ".joblib"]
    
    # Request Timeouts (seconds)
    HEALTH_CHECK_TIMEOUT: float = 5.0
    API_TIMEOUT: float = 30.0
    MODEL_INVOKE_TIMEOUT: float = 60.0
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: str = "60/minute"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
