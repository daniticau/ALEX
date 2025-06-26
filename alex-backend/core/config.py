"""
Configuration management for ALEX Backend
"""

from pydantic import BaseSettings, Field
from typing import Dict, Any, Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Basic settings
    app_name: str = "ALEX"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database
    database_url: str = Field(default="sqlite:///./alex.db", env="DATABASE_URL")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    deepseek_api_key: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    
    # Local model settings
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    ollama_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT")
    
    # Default model settings
    default_model: str = Field(default="ollama/llama3.1:8b", env="DEFAULT_MODEL")
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=2048, env="DEFAULT_MAX_TOKENS")
    
    # Model configurations
    model_configs: Dict[str, Dict[str, Any]] = {
        "ollama/llama3.1:8b": {
            "type": "ollama",
            "model_name": "llama3.1:8b",
            "max_tokens": 4096,
            "temperature": 0.7,
            "system_prompt": "You are ALEX, a helpful AI assistant."
        },
        "ollama/deepseek-coder": {
            "type": "ollama", 
            "model_name": "deepseek-coder:6.7b",
            "max_tokens": 4096,
            "temperature": 0.3,
            "system_prompt": "You are ALEX, a coding assistant. Provide clear, well-commented code."
        },
        "openai/gpt-4": {
            "type": "openai",
            "model_name": "gpt-4",
            "max_tokens": 4096,
            "temperature": 0.7,
            "system_prompt": "You are ALEX, a helpful AI assistant."
        },
        "openai/gpt-3.5-turbo": {
            "type": "openai",
            "model_name": "gpt-3.5-turbo",
            "max_tokens": 4096,
            "temperature": 0.7,
            "system_prompt": "You are ALEX, a helpful AI assistant."
        },
        "anthropic/claude-3-sonnet": {
            "type": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
            "max_tokens": 4096,
            "temperature": 0.7,
            "system_prompt": "You are ALEX, a helpful AI assistant."
        },
        "deepseek/deepseek-chat": {
            "type": "deepseek",
            "model_name": "deepseek-chat",
            "max_tokens": 4096,
            "temperature": 0.7,
            "system_prompt": "You are ALEX, a helpful AI assistant."
        }
    }
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # Security
    cors_origins: list = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Advanced features
    enable_rag: bool = Field(default=False, env="ENABLE_RAG")
    vector_db_url: str = Field(default="./vectordb", env="VECTOR_DB_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    settings = get_settings()
    return settings.model_configs.get(model_name, {})

def list_available_models() -> list:
    """List all configured models"""
    settings = get_settings()
    return list(settings.model_configs.keys())

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider"""
    settings = get_settings()
    key_mapping = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "deepseek": settings.deepseek_api_key
    }
    return key_mapping.get(provider)