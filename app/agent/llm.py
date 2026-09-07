"""The one call every agent node makes into the LLM.

Narrowing to a single `complete()` method — instead of handing nodes the raw
Groq client — is what makes each node testable with a scripted fake instead of
a real API key.
"""

from typing import Protocol

import groq


class LLM(Protocol):
    def complete(self, *, system: str, user: str, json_mode: bool = False) -> str: ...


class GroqLLM:
    """Wraps one Groq chat-completion call, OpenAI-compatible under the hood."""

    def __init__(
        self,
        client: groq.Groq,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return response.choices[0].message.content or ""
