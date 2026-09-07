"""Wires the five nodes into the LangGraph pipeline.

    START -> resolve_destination -> [flight_agent, hotel_agent] (parallel)
          -> itinerary_agent -> final_response_agent -> END

`resolve_destination` only does real work when the traveller left the
destination open; flight and hotel search run in parallel since neither
depends on the other, then join into the itinerary agent, which needs both.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.llm import LLM
from app.agent.nodes import (
    build_final_response_node,
    build_flight_node,
    build_hotel_node,
    build_itinerary_node,
    build_resolve_destination_node,
)
from app.agent.state import TravelState
from app.providers.base import TravelProvider


def build_planner_graph(
    provider: TravelProvider, llm: LLM, max_itinerary_retries: int = 2
) -> CompiledStateGraph:
    graph = StateGraph(TravelState)

    graph.add_node("resolve_destination", build_resolve_destination_node(provider))
    graph.add_node("flight_agent", build_flight_node(provider, llm))
    graph.add_node("hotel_agent", build_hotel_node(provider, llm))
    graph.add_node("itinerary_agent", build_itinerary_node(provider, llm, max_itinerary_retries))
    graph.add_node("final_response_agent", build_final_response_node(llm))

    graph.add_edge(START, "resolve_destination")
    graph.add_edge("resolve_destination", "flight_agent")
    graph.add_edge("resolve_destination", "hotel_agent")
    graph.add_edge("flight_agent", "itinerary_agent")
    graph.add_edge("hotel_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_response_agent")
    graph.add_edge("final_response_agent", END)

    return graph.compile()
