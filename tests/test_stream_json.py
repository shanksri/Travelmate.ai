"""DayStreamer: each day comes out the moment its closing brace arrives,
however the text is chunked."""

import json

from app.agent.stream_json import DayStreamer

PLAN = {
    "days": [
        {
            "day": 1,
            "date": "2027-01-10",
            "summary": "Fort Kochi {and} the \"Chinese\" nets",
            "city": "Kochi",
            "activities": [{"time": "09:00", "title": "Walk", "description": "Braces } { here"}],
        },
        {"day": 2, "date": "2027-01-11", "summary": "Munnar \\ tea", "city": "Munnar",
         "activities": []},
    ],
    "notes": [{"looks": "like a day"}],
}
TEXT = json.dumps(PLAN)


def _feed(chunks):
    streamer = DayStreamer()
    out = []
    for chunk in chunks:
        out.append(streamer.feed(chunk))
    return out


def test_one_character_at_a_time_yields_each_day_once():
    per_chunk = _feed(TEXT)

    days = [day for batch in per_chunk for day in batch]
    assert days == PLAN["days"]


def test_a_day_is_released_by_the_chunk_holding_its_closing_brace():
    first_day_ends = TEXT.index('"city": "Kochi"') + TEXT[TEXT.index('"city": "Kochi"'):].index(
        "]}"
    ) + 2
    per_chunk = _feed([TEXT[:first_day_ends - 1], TEXT[first_day_ends - 1:first_day_ends],
                       TEXT[first_day_ends:]])

    assert per_chunk[0] == []
    assert [d["day"] for d in per_chunk[1]] == [1]
    assert [d["day"] for d in per_chunk[2]] == [2]


def test_braces_and_escaped_quotes_inside_strings_dont_count():
    days = [day for batch in _feed([TEXT[i:i + 5] for i in range(0, len(TEXT), 5)])
            for day in batch]

    assert days[0]["summary"] == 'Fort Kochi {and} the "Chinese" nets'
    assert days[0]["activities"][0]["description"] == "Braces } { here"
    assert days[1]["summary"] == "Munnar \\ tea"


def test_nothing_after_the_days_array_is_taken_for_a_day():
    days = [day for batch in _feed([TEXT]) for day in batch]

    assert len(days) == 2  # the object in `notes` isn't one


def test_text_before_days_waits_for_the_array_to_start():
    streamer = DayStreamer()

    assert streamer.feed('{"da') == []
    assert streamer.feed('ys": [') == []
    assert streamer.feed('{"day": 1}') == [{"day": 1}]
