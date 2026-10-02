"""Pick finished days out of an itinerary while the model is still writing it.

The itinerary arrives as one JSON object, `{"days": [{...}, {...}, ...],
"notes": [...]}` (strict structured outputs, so `days` comes first and the
JSON is well-formed). Streaming it token by token means the page can show
day 1 as soon as its closing brace arrives, instead of waiting ~20 s for the
whole plan.

`DayStreamer` is fed the raw text chunks and returns each element of the
`days` array the moment it's complete. It tracks only what it needs —
whether it's inside a string (so braces in an activity description don't
count), escapes, and brace depth — rather than parsing the JSON for real,
which can't be done until the end.
"""

import json
import re

_DAYS_ARRAY = re.compile(r'"days"\s*:\s*\[')


class DayStreamer:
    def __init__(self) -> None:
        self._text = ""
        self._pos: int | None = None  # where scanning resumes, once inside `days`
        self._in_string = False
        self._escaped = False
        self._depth = 0  # depth within the days array: 1 = inside one day object
        self._day_start: int | None = None
        self._finished = False

    def feed(self, chunk: str) -> list[dict]:
        """Add a chunk of the model's output; return any days it completed."""
        if self._finished:
            return []
        self._text += chunk
        if self._pos is None:
            match = _DAYS_ARRAY.search(self._text)
            if match is None:
                return []
            self._pos = match.end()

        completed: list[dict] = []
        text = self._text
        i = self._pos
        while i < len(text):
            char = text[i]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char == "{":
                if self._depth == 0:
                    self._day_start = i
                self._depth += 1
            elif char == "}":
                self._depth -= 1
                if self._depth == 0 and self._day_start is not None:
                    try:
                        completed.append(json.loads(text[self._day_start : i + 1]))
                    except json.JSONDecodeError:
                        pass  # not ours to judge — the full parse afterwards will
                    self._day_start = None
            elif char == "]" and self._depth == 0:
                self._finished = True
                i += 1
                break
            i += 1
        self._pos = i
        return completed
