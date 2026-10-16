"""The eval suite's own plumbing: the cases are well formed, and a case is
judged by its checks. The evals themselves run against the real model
(evals/run.py), never in this suite."""

import json

from conftest import FakeLLM

from app.core.config import Settings
from app.providers.mock import MockTravelProvider
from evals import run
from evals.cases import CASES, Case, Check, refused


def test_every_case_is_well_formed():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))
    for case in CASES:
        assert case.checks, case.name
        assert case.kind in {"interpret", "route", "trip", "revise"}, case.name
        assert case.group in {"guardrail", "parsing", "route", "trip", "revision"}, case.name
        assert bool(case.changes) == (case.kind == "revise"), case.name


def _with_fake_llm(monkeypatch, answer: dict):
    llm = FakeLLM(responses=[json.dumps(answer)])
    monkeypatch.setattr(run, "_llms", lambda settings, usage: (llm, llm, llm))


def test_a_case_passes_when_every_check_holds(monkeypatch):
    _with_fake_llm(monkeypatch, {"verdict": "not_travel", "concerns": ["A poem."]})
    case = Case("poem", "guardrail", "interpret", "write a poem", [refused("not_travel")])

    result = run.run_case(case, Settings(), MockTravelProvider())

    assert result["outcome"] == "pass" and result["failed_checks"] == []


def test_a_case_fails_with_the_checks_that_did_not_hold(monkeypatch):
    _with_fake_llm(monkeypatch, {"verdict": "ok", "concerns": [], "destination": "Goa"})
    case = Case(
        "poem", "guardrail", "interpret", "write a poem",
        [refused("not_travel"), Check("crashes", lambda r: r.nothing.here)],
    )

    result = run.run_case(case, Settings(), MockTravelProvider())

    assert result["outcome"] == "fail"
    assert result["failed_checks"][0] == "refused as not_travel"
    assert result["failed_checks"][1].startswith("crashes (AttributeError")


def test_cost_is_worked_out_from_tokens():
    usage = run.Usage()
    usage.add("gpt-4o-mini-2024-07-18", type("U", (), {"prompt_tokens": 1_000_000,
                                                        "completion_tokens": 0})())
    usage.add("gpt-4o", type("U", (), {"prompt_tokens": 0, "completion_tokens": 100_000})())

    assert usage.calls == 2
    assert round(usage.dollars, 2) == round(0.15 + 1.00, 2)
