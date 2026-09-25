"""Run the support-ticket benchmark through TypeSafe (Jev) and Laya, in every wording.

Run:  uv run --env-file .env --extra typesafe --extra laya python benchmarks/run.py
Makes 63 TypeSafe API calls (3 wordings x (1 warm-up + 20 cases)).
"""

import statistics

import laya
from typesafe_sdk import TypeSafeClient

from semantic_operators import Provider
from semantic_operators.bench import run, stability
from semantic_operators.providers.laya import Laya
from semantic_operators.providers.typesafe import TypeSafe
from support_tickets import cases, wordings


def benchmark(name: str, provider: Provider) -> None:
    reports = {wording: run(provider, qs, cases) for wording, qs in wordings.items()}
    stable = stability(list(reports.values()))
    latencies = [ms for r in reports.values() for ms in r.latencies_ms]

    print(f"\n{name}: median {statistics.median(latencies):.0f} ms per call")
    print("  accuracy, p(correct) in brackets")
    print(f"  {'question':<14}" + "".join(f"{w:>18}" for w in reports) + f"{'stability':>12}")
    for question in stable:
        cells = "".join(
            f"{f'{s.correct}/{s.total} ({s.mean_p_correct:.2f})':>18}"
            for s in (r.questions[question] for r in reports.values())
        )
        print(f"  {question:<14}{cells}{stable[question]:>12.0%}")


with TypeSafeClient() as client:
    benchmark("TypeSafe (jev-latest)", TypeSafe(client))

benchmark("Laya", Laya(laya.load("convaiinnovations/laya")))
