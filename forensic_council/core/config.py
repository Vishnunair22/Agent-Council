"""
Configuration Management Module
===============================

Pydantic Settings-based configuration loading from environment variables.
All configuration is centralized and validated at startup.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via .env file or environment variables.
    Environment variables take precedence over .env file values.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application Settings
    app_name: str = Field(default="forensic_council", description="Application name")
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Debug mode flag")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis server host")
    redis_port: int = Field(default=6379, description="Redis server port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant REST API port")
    qdrant_grpc_port: int = Field(default=6334, description="Qdrant gRPC port")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    
    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", description="PostgreSQL server host")
    postgres_port: int = Field(default=5432, description="PostgreSQL server port")
    postgres_user: str = Field(default="forensic_user", description="PostgreSQL username")
    postgres_password: str = Field(default="forensic_pass", description="PostgreSQL password")
    postgres_db: str = Field(default="forensic_council", description="PostgreSQL database name")
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL from components."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model to use")
    
    # LangChain Configuration
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangChain tracing")
    langchain_api_key: Optional[str] = Field(default=None, description="LangChain API key")
    langchain_project: str = Field(default="forensic_council", description="LangChain project name")
    
    # Storage Configuration
    evidence_storage_path: str = Field(default="./storage/evidence", description="Path for evidence storage")
    calibration_models_path: str = Field(default="./storage/calibration_models", description="Path for calibration models")
    
    # Security Configuration
    signing_key: str = Field(default="change-me-in-production", description="Key for signing audit entries")
    
    # Agent Configuration
    default_iteration_ceiling: int = Field(default=20, description="Default iteration ceiling for agent loops")
    hitl_enabled: bool = Field(default=True, description="Enable Human-in-the-Loop checkpoints")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}, got {v}")
        return v_upper
    
    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Validate application environment."""
        allowed = ["development", "staging", "production", "testing"]
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Environment must be one of {allowed}, got {v}")
        return v_lower


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to ensure settings are only loaded once.
    Call settings.cache_clear() to reload settings (useful for testing).
    """
    return Settings()


# Convenience alias
settings = get_settings()
