from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health, trips
from app.core.logging import configure_logging
from app.store import get_store

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the store (and, for SqlTripStore, create its table) at boot
    # rather than on the first request — a bad TRAVELMATE_DATABASE_URL should
    # fail the container immediately, not the first trip a user plans.
    get_store()
    yield


app = FastAPI(
    title="travelmate.ai",
    version="0.1.0",
    description="TripMate AI: a LangGraph multi-agent travel planner backed by OpenAI.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(trips.router)
