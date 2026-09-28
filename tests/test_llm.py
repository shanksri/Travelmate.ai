"""Structured outputs: the schemas sent to OpenAI must meet strict mode's
rules, and every JSON call must send one. No real API call is made here —
the schemas were checked against the live API once, when this landed."""

import json
from types import SimpleNamespace

import pytest

from app.agent.llm import OpenAILLM, strict_schema
from app.agent.prompt_parser import ParsedPrompt
from app.agent.reviser import ChangeRoute
from app.models.itinerary import DraftItinerary

OUTPUT_MODELS = [ParsedPrompt, DraftItinerary, ChangeRoute]


def _objects(node):
    """Every object schema inside `node`, however deeply nested."""
    if isinstance(node, dict):
        if isinstance(node.get("properties"), dict):
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _objects(item)


def _all_nodes(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _all_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _all_nodes(item)


@pytest.mark.parametrize("model", OUTPUT_MODELS, ids=lambda m: m.__name__)
def test_every_object_lists_all_its_fields_as_required_and_allows_no_others(model):
    schema = strict_schema(model)

    objects = list(_objects(schema))
    assert objects, "expected at least the root object"
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])


@pytest.mark.parametrize("model", OUTPUT_MODELS, ids=lambda m: m.__name__)
def test_no_defaults_and_nothing_beside_a_ref(model):
    for node in _all_nodes(strict_schema(model)):
        # A field literally named "default" would be a property, not a keyword.
        assert "default" not in node or isinstance(node.get("default"), dict)
        if "$ref" in node:
            assert list(node) == ["$ref"]


def test_optional_fields_stay_nullable():
    """Strict mode requires every field, so "optional" has to mean "may be
    null" — the model must still be able to leave a destination unset."""
    destination = strict_schema(ParsedPrompt)["properties"]["destination"]

    assert {"type": "null"} in destination["anyOf"]


def test_the_route_schema_uses_the_return_key_the_prompt_asks_for():
    assert "return" in strict_schema(ChangeRoute)["properties"]


def test_the_model_is_not_changed_by_building_its_schema():
    before = json.dumps(DraftItinerary.model_json_schema(), sort_keys=True)
    strict_schema(DraftItinerary)
    assert json.dumps(DraftItinerary.model_json_schema(), sort_keys=True) == before


# --- the OpenAI wrapper --------------------------------------------------------


class _FakeClient:
    def __init__(self, message):
        self.sent: dict = {}
        self._message = message
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


def _llm(message) -> tuple[OpenAILLM, _FakeClient]:
    client = _FakeClient(message)
    return OpenAILLM(client, model="gpt-4o-mini"), client


def test_a_schema_is_sent_as_a_strict_json_schema():
    llm, client = _llm(SimpleNamespace(content='{"days": []}', refusal=None))

    llm.complete(system="s", user="u", json_mode=True, schema=DraftItinerary)

    fmt = client.sent["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "DraftItinerary"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == strict_schema(DraftItinerary)


def test_json_mode_alone_still_asks_for_a_json_object():
    llm, client = _llm(SimpleNamespace(content="{}", refusal=None))

    llm.complete(system="s", user="u", json_mode=True)

    assert client.sent["response_format"] == {"type": "json_object"}


def test_plain_text_calls_send_no_format():
    llm, client = _llm(SimpleNamespace(content="A lovely trip.", refusal=None))

    assert llm.complete(system="s", user="u") == "A lovely trip."
    assert "response_format" not in client.sent


def test_a_refusal_comes_back_empty_for_the_usual_retry_path():
    llm, _ = _llm(SimpleNamespace(content=None, refusal="I can't help with that."))

    assert llm.complete(system="s", user="u", schema=ParsedPrompt) == ""


# --- every JSON call sends its schema -----------------------------------------


def test_the_parser_sends_its_schema():
    from conftest import FakeLLM

    from app.agent.prompt_parser import parse_trip_prompt

    llm = FakeLLM(responses=[json.dumps({"destination": "Goa"})])
    parse_trip_prompt("Goa", llm)

    assert llm.calls[0]["schema"] is ParsedPrompt


def test_the_itinerary_agent_sends_its_schema(provider, trip_request):
    from conftest import FakeLLM, draft_itinerary_json

    from app.agent.nodes import build_itinerary_node
    from app.agent.state import initial_state

    llm = FakeLLM(responses=[draft_itinerary_json(trip_request)])
    node = build_itinerary_node(provider, llm, max_retries=0)
    state = initial_state(trip_request) | {
        "resolved_destination": "Lisbon, Portugal",
        "flight_results": {},
        "hotel_results": {},
    }

    node(state)

    assert llm.calls[0]["schema"] is DraftItinerary
