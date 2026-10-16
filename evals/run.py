"""Run the evals against the real model and report what passed.

    docker compose --profile evals run --rm evals                 # every case
    docker compose --profile evals run --rm evals --only guardrail
    docker compose --profile evals run --rm evals --case poem --repeat 5
    docker compose --profile evals run --rm evals --live          # real flights/hotels

Flights and hotels come from the mock provider unless --live: the evals test
what the model does, and live searches would spend SerpApi quota on every run.
Everything else is real: OpenAI calls (the planning model, gpt-4o for routes
and revisions), and the airport list the route checks use.

Prints one line per case and a summary per group, with OpenAI tokens and an
approximate cost, and writes the whole run to evals/results/<time>.json.
Exits 1 if any case without a `known_issue` failed, so it can gate a change.
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# US dollars per million tokens (input, output), at list prices when this was
# written. Only for the cost estimate; update if they change.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}

RESULTS_DIR = Path(__file__).parent / "results"


# --- counting what each case spends --------------------------------------------


class Usage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.by_model: dict[str, list[int]] = {}

    def add(self, model: str, usage) -> None:
        if usage is None:
            return
        with self.lock:
            totals = self.by_model.setdefault(model, [0, 0, 0])
            totals[0] += 1
            totals[1] += usage.prompt_tokens
            totals[2] += usage.completion_tokens

    @property
    def calls(self) -> int:
        return sum(t[0] for t in self.by_model.values())

    @property
    def dollars(self) -> float:
        total = 0.0
        for model, (_, tokens_in, tokens_out) in self.by_model.items():
            price_in, price_out = next(
                (p for name, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0]))
                 if model.startswith(name)),
                (0.0, 0.0),
            )
            total += tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out
        return total


class _Completions:
    """`client.chat.completions`, noting each call's tokens."""

    def __init__(self, inner, usage: Usage) -> None:
        self._inner, self._usage = inner, usage

    def create(self, **kwargs):
        if kwargs.get("stream"):
            kwargs["stream_options"] = {"include_usage": True}
            return self._counted_stream(self._inner.create(**kwargs), kwargs["model"])
        response = self._inner.create(**kwargs)
        self._usage.add(kwargs["model"], response.usage)
        return response

    def _counted_stream(self, stream, model):
        for chunk in stream:
            if getattr(chunk, "usage", None):
                self._usage.add(model, chunk.usage)
            yield chunk


class CountingClient:
    def __init__(self, inner, usage: Usage) -> None:
        self.chat = SimpleNamespace(completions=_Completions(inner.chat.completions, usage))


# --- running one case ------------------------------------------------------------


def _llms(settings, usage: Usage):
    import openai

    from app.agent.llm import OpenAILLM

    client = CountingClient(openai.OpenAI(), usage)
    planning = OpenAILLM(client, settings.model, settings.temperature, settings.max_tokens)
    return planning, planning.with_model(settings.revise_model), planning.with_model(
        settings.route_model
    )


def _interpret(case, llm):
    from app.agent.prompt_parser import RequestRefused, interpret_prompt

    try:
        found = interpret_prompt(case.prompt, llm)
    except RequestRefused as exc:
        return SimpleNamespace(refused=exc.verdict, message=str(exc), intent=None,
                               parsed=None, trip=None)
    return SimpleNamespace(refused=None, message="", intent=found.intent,
                           parsed=found.parsed, trip=found.trip)


def _route(case, llm, route_llm, provider):
    from app.agent.route_plan import plan_route
    from app.providers.iata import resolve_iata

    result = _interpret(case, llm)
    result.route = None
    if result.trip is None:
        return result
    request = result.trip.model_copy(update={"include_flights": True})
    result.route, result.note = plan_route(request, request.destination, route_llm, provider)
    result.airports_found = all(
        resolve_iata(city) for city in (result.route.fly_into, result.route.fly_out_of)
    )
    return result


def _trip(case, llm, settings, provider):
    from app.agent.assistant import answer_prompt
    from app.agent.nodes import _base_by_date

    answer = answer_prompt(
        case.prompt, llm=llm, provider=provider, settings=settings,
        include_flights=case.flights, include_hotels=case.hotels,
    )
    trip = answer.trip
    bases = list(_base_by_date(trip.request, trip.itinerary.stays).values()) if trip else []
    return SimpleNamespace(trip=trip, route=None, bases=bases)


def _revise(case, llm, revise_llm, settings, provider):
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.memory import _serializer
    from app.agent.reviser import RevisionDeclined, revise_trip

    base = _trip(case, llm, settings, provider).trip
    memory = InMemorySaver(serde=_serializer())
    steps, current = [], base
    for change in case.changes:
        try:
            current = revise_trip(current, change, llm=revise_llm, settings=settings,
                                  checkpointer=memory)
            steps.append(SimpleNamespace(change=change, trip=current, declined=None))
        except RevisionDeclined as exc:
            steps.append(SimpleNamespace(change=change, trip=None, declined=str(exc)))
    return SimpleNamespace(base=base, steps=steps)


def run_case(case, settings, provider) -> dict:
    usage = Usage()
    llm, revise_llm, route_llm = _llms(settings, usage)
    started = time.monotonic()
    failed, error = [], None
    try:
        if case.kind == "interpret":
            result = _interpret(case, llm)
        elif case.kind == "route":
            result = _route(case, llm, route_llm, provider)
        elif case.kind == "trip":
            result = _trip(case, llm, settings, provider)
        else:
            result = _revise(case, llm, revise_llm, settings, provider)
        for check in case.checks:
            try:
                ok = bool(check.holds(result))
            except Exception as exc:  # a check that can't be evaluated has failed
                ok = False
                failed.append(f"{check.says} ({type(exc).__name__}: {exc})")
                continue
            if not ok:
                failed.append(check.says)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return {
        "case": case.name,
        "group": case.group,
        "outcome": "error" if error else ("fail" if failed else "pass"),
        "failed_checks": failed,
        "error": error,
        "known_issue": case.known_issue,
        "seconds": round(time.monotonic() - started, 1),
        "calls": usage.calls,
        "tokens": {m: {"calls": c, "in": i, "out": o} for m, (c, i, o) in usage.by_model.items()},
        "dollars": round(usage.dollars, 5),
    }


# --- the report ------------------------------------------------------------------


def _mark(run: dict) -> str:
    if run["outcome"] == "pass":
        return "PASS"
    return "KNOWN" if run["known_issue"] else run["outcome"].upper()


def report(runs: list[dict], cases_by_name: dict, wall: float) -> bool:
    by_case: dict[str, list[dict]] = {}
    for run in runs:
        by_case.setdefault(run["case"], []).append(run)

    print()
    for name, case_runs in by_case.items():
        passed = sum(r["outcome"] == "pass" for r in case_runs)
        first_bad = next((r for r in case_runs if r["outcome"] != "pass"), None)
        mark = "PASS" if first_bad is None else _mark(first_bad)
        rate = f"{passed}/{len(case_runs)}" if len(case_runs) > 1 else ""
        seconds = sum(r["seconds"] for r in case_runs) / len(case_runs)
        print(f"{mark:5} {cases_by_name[name].group:9} {name:30} {rate:5} {seconds:5.1f}s")
        if first_bad:
            for why in first_bad["failed_checks"]:
                print(f"        ✗ {why}")
            if first_bad["error"]:
                print(f"        ! {first_bad['error']}")
            if first_bad["known_issue"]:
                print(f"        (known: {first_bad['known_issue']})")

    print()
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["group"], []).append(run)
    for group, group_runs in groups.items():
        passed = sum(r["outcome"] == "pass" for r in group_runs)
        print(f"{group:10} {passed}/{len(group_runs)} passed")

    regressions = [r for r in runs if r["outcome"] != "pass" and not r["known_issue"]]
    calls = sum(r["calls"] for r in runs)
    dollars = sum(r["dollars"] for r in runs)
    print(
        f"\n{sum(r['outcome'] == 'pass' for r in runs)}/{len(runs)} passed, "
        f"{len(regressions)} unexpected failure(s). {calls} OpenAI calls, "
        f"about ${dollars:.3f}, {wall:.0f}s."
    )
    return not regressions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help="comma-separated groups to run")
    parser.add_argument("--case", action="append", help="run just this case (repeatable)")
    parser.add_argument("--repeat", type=int, default=1, help="run each case N times")
    parser.add_argument("--live", action="store_true", help="real flights and hotels")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    # Before any app module reads its settings.
    os.environ["TRAVELMATE_PROVIDER"] = "live" if args.live else "mock"
    os.environ.setdefault("TRAVELMATE_STORE", "memory")
    os.environ.setdefault("LANGSMITH_TRACING", "false")

    from app.core.config import get_settings
    from app.providers import get_provider
    from evals.cases import CASES

    cases = CASES
    if args.only:
        groups = set(args.only.split(","))
        cases = [c for c in cases if c.group in groups]
    if args.case:
        cases = [c for c in cases if c.name in args.case]
    if not cases:
        print("No cases match.")
        return 2

    settings, provider = get_settings(), get_provider()
    print(
        f"{len(cases)} case(s) x {args.repeat}, model {settings.model}, routes "
        f"{settings.route_model}, revisions {settings.revise_model}, "
        f"provider {settings.provider}"
    )
    started = time.monotonic()
    jobs = [case for case in cases for _ in range(args.repeat)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(lambda c: run_case(c, settings, provider), jobs))
    wall = time.monotonic() - started

    ok = report(runs, {c.name: c for c in cases}, wall)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps(
            {"settings": {"model": settings.model, "route_model": settings.route_model,
                          "revise_model": settings.revise_model,
                          "provider": settings.provider},
             "repeat": args.repeat, "runs": runs},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
