"""Errors raised by graph steps, in one place so the steps, the planner and
the API can all import them without importing each other."""


class PlanningError(RuntimeError):
    """The agent pipeline ran but never produced a usable itinerary."""
