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

    # Revisions only (app/agent/reviser.py). Fitting a new place into a
    # finished route means judging travel distances, which gpt-4o-mini gets
    # wrong: it agreed to "3 days in Ladakh" on a Kochi trip by leaving the
    # traveller in Leh on departure day. gpt-4o declined it correctly. Same
    # JSON-mode requirement as `model`.
    revise_model: str = "gpt-4o"
    temperature: float = 0.3

    # OpenAI's JSON mode guarantees syntactically valid JSON *except* when the
    # response gets cut off by hitting this ceiling mid-generation — a longer
    # or packed-pace itinerary can genuinely need more than a few thousand
    # tokens. 16000 leaves real headroom under gpt-4o-mini's 16384-token cap
    # without changing cost: OpenAI bills by tokens actually generated, not
    # this ceiling.
    max_tokens: int = 16000

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

    # How long a live flight search is reused before searching again. Every
    # search spends SerpApi quota (100/month on the free plan), so re-planning
    # the same route and date within this window costs nothing. 0 disables it.
    flight_cache_ttl_hours: float = 6.0

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
