import json

from conftest import itinerary_payload

from app.agent.tools import ItinerarySink, build_research_tools, build_submit_tool


def tools_by_name(provider, trip_request) -> dict:
    return {fn.__name__: fn for fn in build_research_tools(provider, trip_request)}


def test_research_tools_return_json(provider, trip_request):
    tools = tools_by_name(provider, trip_request)
    out = json.loads(tools["search_attractions"]("Lisbon, Portugal", "food", 5))
    assert len(out["attractions"]) == 5


def test_flight_search_uses_the_request_party_size(provider, trip_request):
    tools = tools_by_name(provider, trip_request)
    out = json.loads(tools["search_flights"]("BOS", "LIS", "2026-04-10"))
    assert out["travelers"] == trip_request.travelers


def test_bad_date_comes_back_as_a_correctable_error(provider, trip_request):
    tools = tools_by_name(provider, trip_request)
    out = json.loads(tools["search_flights"]("BOS", "LIS", "April 10th"))
    assert "error" in out


def test_bad_budget_level_is_rejected(provider, trip_request):
    tools = tools_by_name(provider, trip_request)
    out = json.loads(tools["search_destinations"]("food", "extravagant"))
    assert "error" in out


def test_submit_accepts_a_valid_itinerary(trip_request):
    sink = ItinerarySink()
    submit = build_submit_tool(sink, trip_request)

    result = json.loads(submit(json.dumps(itinerary_payload(trip_request))))

    assert result["ok"] is True
    assert sink.received
    assert sink.itinerary.destination == "Lisbon, Portugal"
    assert len(sink.itinerary.days) == trip_request.nights + 1


def test_submit_rejects_malformed_json(trip_request):
    sink = ItinerarySink()
    result = json.loads(build_submit_tool(sink, trip_request)("{not json"))

    assert result["ok"] is False
    assert "expected_shape" in result
    assert not sink.received


def test_submit_reports_validation_problems(trip_request):
    sink = ItinerarySink()
    payload = itinerary_payload(trip_request)
    del payload["destination"]

    result = json.loads(build_submit_tool(sink, trip_request)(json.dumps(payload)))

    assert result["ok"] is False
    assert any(p["field"] == "destination" for p in result["problems"])
    assert not sink.received


def test_submit_rejects_a_day_count_that_does_not_match_the_dates(trip_request):
    sink = ItinerarySink()
    payload = itinerary_payload(trip_request)
    payload["days"] = payload["days"][:2]

    result = json.loads(build_submit_tool(sink, trip_request)(json.dumps(payload)))

    assert result["ok"] is False
    assert "days" in result["error"]
    assert not sink.received
