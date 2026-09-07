from fastapi import FastAPI

from app.api.routes import health, trips
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="travelmate.ai",
    version="0.1.0",
    description=(
        "TripMate AI: a LangGraph multi-agent travel planner backed by Groq's Llama 3."
    ),
)

app.include_router(health.router)
app.include_router(trips.router)
