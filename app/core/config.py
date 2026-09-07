"""Application settings, loaded from the environment or a local .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TRAVELMATE_", extra="ignore"
    )

    # The OpenAI model every agent node reasons with. Must support JSON mode
    # (response_format={"type": "json_object"}) — the itinerary agent depends
    # on it; plain "gpt-4" does not support it and fails with a 400.
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 4096

    # How many times the itinerary agent may retry after a schema-invalid
    # response before the run fails outright.
    max_itinerary_retries: int = 2

    # "mock" serves deterministic travel data so the agent runs without any
    # third-party keys. "live" swaps in real provider clients.
    provider: Literal["mock", "live"] = "mock"

    # "memory" needs no setup and is what the tests run against. "postgres"
    # persists trips across restarts — see docker-compose.yml for a ready-made
    # Postgres service, and app/store.py for the SqlTripStore it talks to.
    store: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql+psycopg://travelmate:travelmate@localhost:5433/travelmate"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
