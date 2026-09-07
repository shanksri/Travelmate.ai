"""Application settings, loaded from the environment or a local .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TRAVELMATE_", extra="ignore"
    )

    # Which Claude model plans the trip, and how hard it is allowed to think.
    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = 16000

    # Safety rail on the agentic loop: how many assistant turns before we stop.
    max_tool_iterations: int = 24

    # "mock" serves deterministic travel data so the agent runs without any
    # third-party keys. "live" swaps in real provider clients.
    provider: Literal["mock", "live"] = "mock"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
