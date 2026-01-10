from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "Record Linkage App"
    debug: bool = Field(default=False, env="DEBUG")

    # Security
    secret_key: str = Field(
        ...,  # Required, no default!
        env="SECRET_KEY",
        description="Secret key for JWT - MUST be set"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(
        default=30,  # 30 minutes
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Database
    database_url: str = "sqlite:///./storage/app.db"

    # Storage
    storage_path: str = "storage"
    upload_dir: str = "storage/uploads"
    models_dir: str = "storage/models"
    max_upload_size_mb: int = 100

    # Model security
    model_secret_key: str = Field(
        ...,  # Required, no default!
        env="MODEL_SECRET_KEY",
        description="Secret key for model signing - MUST be set"
    )

    # Base directory
    base_dir: Path = Path(__file__).resolve().parent.parent

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
