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


DEFAULT_TOM_SYSTEM_PROMPT = (
    "You are TOM, a personal AI assistant. "
    "Never identify yourself as Qwen or mention the underlying model. "
    "Communicate in a helpful, precise, and natural manner. "
    "Do not claim to have performed actions you did not actually perform. "
    "Answer simple questions concisely and provide detail when requested."
)


class Config:
    """Core application configuration."""

    APP_NAME: str = os.getenv("APP_NAME", "TOM")
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    TOM_SYSTEM_PROMPT: str = os.getenv("TOM_SYSTEM_PROMPT", DEFAULT_TOM_SYSTEM_PROMPT)
    MAX_CONVERSATION_MESSAGES: int = int(os.getenv("MAX_CONVERSATION_MESSAGES", "20"))


config = Config()
TOM_SYSTEM_PROMPT = config.TOM_SYSTEM_PROMPT
