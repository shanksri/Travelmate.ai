"""Application settings, loaded from the environment or a local .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TRAVELMATE_", extra="ignore"
    )

    # The Groq-hosted Llama 3 model every agent node reasons with.
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 4096

    # How many times the itinerary agent may retry after a schema-invalid
    # response before the run fails outright.
    max_itinerary_retries: int = 2

    # "mock" serves deterministic travel data so the agent runs without any
    # third-party keys. "live" swaps in real provider clients.
    provider: Literal["mock", "live"] = "mock"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
