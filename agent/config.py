from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.2

    max_steps: int = Field(default=6, ge=1, le=20)
    history_messages: int = Field(default=10, ge=0)
    tasks_db_path: str = "data/tasks.db"
