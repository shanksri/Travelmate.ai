from fastapi import FastAPI

from app.api.routes import health, trips
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="travelmate.ai",
    version="0.1.0",
    description="An agentic trip planner: Claude plans itineraries by calling travel tools.",
)

app.include_router(health.router)
app.include_router(trips.router)
