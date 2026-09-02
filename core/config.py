"""Configuration module for TOM."""
import os
from pathlib import Path


def load_env_file(dotenv_path: Path | None = None) -> None:
    """Simple .env file loader without external dependencies."""
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"

    if not dotenv_path.is_file():
        return

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            os.environ.setdefault(key, val)


# Load environment variables on module import
load_env_file()


class Config:
    """Core application configuration."""

    APP_NAME: str = os.getenv("APP_NAME", "TOM")
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")


config = Config()
