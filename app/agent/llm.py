"""The one call every agent node makes into the LLM.

Narrowing to a single `complete()` method — instead of handing nodes the raw
OpenAI client — is what makes each node testable with a scripted fake instead
of a real API key.

Calls that expect JSON pass `schema`, the Pydantic model the answer must fit.
OpenAI's structured outputs (`json_schema`, strict) then make it impossible
for the model to return anything else: no missing or misspelled keys, no
prose around the JSON. Before this, those calls used JSON mode, which only
guarantees *some* JSON object, and every shape mistake cost a retry — a full
LLM call. Checks a schema can't express (exact trip dates, a change that
changed nothing) still run afterwards.
"""

import logging
from collections.abc import Iterator
from typing import Any, Protocol

import openai
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLM(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        json_mode: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> str: ...


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """`model`'s JSON schema, adjusted to what OpenAI's strict mode accepts:
    every property listed as required (optional ones stay nullable), no
    additional properties, no `default`s, and no keywords beside a `$ref`."""
    schema = model.model_json_schema()

    def fix(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                fix(item)
            return
        if not isinstance(node, dict):
            return
        node.pop("default", None)
        if "$ref" in node:
            for key in [k for k in node if k != "$ref"]:
                del node[key]
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties)
            for field_schema in properties.values():
                fix(field_schema)
        for key, value in node.items():
            if key != "properties":
                fix(value)

    fix(schema)
    return schema


class OpenAILLM:
    """Wraps one OpenAI chat-completion call."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 16000,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _request(
        self, system: str, user: str, json_mode: bool, schema: type[BaseModel] | None
    ) -> dict:
        kwargs: dict = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": strict_schema(schema),
                    "strict": True,
                },
            }
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def stream(
        self,
        *,
        system: str,
        user: str,
        json_mode: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> Iterator[str]:
        """The same call as `complete`, yielding the answer's text as it's
        generated. Used by the itinerary agent so the page can show each day
        as soon as it's written (app/agent/stream_json.py)."""
        response = self._client.chat.completions.create(
            stream=True, **self._request(system, user, json_mode, schema)
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "refusal", None):
                logger.warning("model refused: %s", delta.refusal)
            if delta.content:
                yield delta.content

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_mode: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            **self._request(system, user, json_mode, schema)
        )
        message = response.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            # Structured outputs report a safety refusal here instead of in
            # the content. Returning "" lets the caller's usual validation
            # and retry path handle it like any unusable answer.
            logger.warning("model refused: %s", refusal)
            return ""
        return message.content or ""
