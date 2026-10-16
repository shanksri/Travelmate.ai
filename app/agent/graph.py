"""The two graphs every request runs through.

The trip graph plans one trip from a `TripRequest`:

    START -> resolve_destination -> plan_route
          -> [flight_agent, hotel_agent] (parallel)
          -> itinerary_agent -> check_budget -> final_response_agent
          -> package_trip -> END

`resolve_destination` only does real work when the traveller left the
destination open. `plan_route` decides the towns slept in and the airports
(app/agent/route_plan.py), so flights search real airports and hotels are
searched per town. Flight and hotel search run in parallel since neither
depends on the other, then join into the itinerary agent, which needs both.
`check_budget` compares the total with the traveller's budget in code
(app/agent/budget.py), before the summary is written. `package_trip` turns
the result into a saveable `PlannedTrip`.

The ask graph is the prompt box. It reads the sentence first and branches:

    START -> interpret -> places ------------------------------------> END
                       -> route -------------------------------------> END
                       -> prepare_trip -> (the trip graph's steps)
                                       -> package_trip -> city_places x N -> END

`city_places` runs once per city of the trip, in parallel (a LangGraph
`Send` fan-out), only when the page's Restaurants box asked for places.

Revisions have a graph of their own, in app/agent/reviser.py.

Every step reports progress through app/agent/events.py, so streaming either
graph (app/jobs.py) shows the page what's happening as it happens.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.ask_nodes import (
    build_city_places_node,
    build_interpret_node,
    build_places_node,
    build_prepare_trip_node,
    build_route_node,
    next_after_interpret,
    places_for_each_city,
)
from app.agent.budget import build_budget_check_node
from app.agent.llm import LLM
from app.agent.nodes import (
    build_final_response_node,
    build_flight_node,
    build_hotel_node,
    build_itinerary_node,
    build_package_trip_node,
    build_resolve_destination_node,
)
from app.agent.route_plan import build_route_plan_node
from app.agent.state import AskState, TravelState
from app.providers.base import TravelProvider


def _add_trip_steps(
    graph: StateGraph,
    provider: TravelProvider,
    llm: LLM,
    max_itinerary_retries: int,
    route_model: str | None,
) -> None:
    """resolve_destination … package_trip, wired together. The caller wires
    what leads into `resolve_destination` and out of `package_trip`."""
    graph.add_node("resolve_destination", build_resolve_destination_node(provider))
    graph.add_node("plan_route", build_route_plan_node(provider, llm, route_model))
    graph.add_node("flight_agent", build_flight_node(provider, llm))
    graph.add_node("hotel_agent", build_hotel_node(provider))
    graph.add_node("itinerary_agent", build_itinerary_node(llm, max_itinerary_retries))
    graph.add_node("check_budget", build_budget_check_node())
    graph.add_node("final_response_agent", build_final_response_node(llm))
    graph.add_node("package_trip", build_package_trip_node())

    graph.add_edge("resolve_destination", "plan_route")
    graph.add_edge("plan_route", "flight_agent")
    graph.add_edge("plan_route", "hotel_agent")
    graph.add_edge("flight_agent", "itinerary_agent")
    graph.add_edge("hotel_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "check_budget")
    graph.add_edge("check_budget", "final_response_agent")
    graph.add_edge("final_response_agent", "package_trip")


def build_planner_graph(
    provider: TravelProvider,
    llm: LLM,
    max_itinerary_retries: int = 2,
    route_model: str | None = None,
) -> CompiledStateGraph:
    graph = StateGraph(TravelState)
    _add_trip_steps(graph, provider, llm, max_itinerary_retries, route_model)
    graph.add_edge(START, "resolve_destination")
    graph.add_edge("package_trip", END)
    return graph.compile()


def build_ask_graph(
    provider: TravelProvider,
    llm: LLM,
    max_itinerary_retries: int = 2,
    route_model: str | None = None,
) -> CompiledStateGraph:
    graph = StateGraph(AskState)

    graph.add_node("interpret", build_interpret_node(llm))
    graph.add_node("places", build_places_node())
    graph.add_node("route", build_route_node())
    graph.add_node("prepare_trip", build_prepare_trip_node())
    _add_trip_steps(graph, provider, llm, max_itinerary_retries, route_model)
    graph.add_node("city_places", build_city_places_node())

    graph.add_edge(START, "interpret")
    graph.add_conditional_edges(
        "interpret", next_after_interpret, ["places", "route", "prepare_trip"]
    )
    graph.add_edge("places", END)
    graph.add_edge("route", END)
    graph.add_edge("prepare_trip", "resolve_destination")
    graph.add_conditional_edges("package_trip", places_for_each_city, ["city_places", END])
    graph.add_edge("city_places", END)

    return graph.compile()
