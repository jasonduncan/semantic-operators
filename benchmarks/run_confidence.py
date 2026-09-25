"""The "don't know" trade-off: stricter min_confidence answers fewer cases, but are
the answers it still gives more often right? That's only useful if the model's
confidence means something, which is exactly what this measures.

Run:  uv run --env-file .env --extra typesafe --extra laya python benchmarks/run_confidence.py
Makes 21 TypeSafe API calls (1 warm-up + 20 cases). Thresholds are applied afterwards
with bench.at_min_confidence, so they cost no extra calls.
"""

import laya
from typesafe_sdk import TypeSafeClient

from semantic_operators import Provider
from semantic_operators.bench import at_min_confidence, run
from semantic_operators.providers.laya import Laya
from semantic_operators.providers.typesafe import TypeSafe
from support_tickets import cases, questions

THRESHOLDS = (0.0, 0.6, 0.8, 0.9)


def benchmark(name: str, provider: Provider) -> None:
    report = run(provider, questions, cases)
    print(f"\n{name}: answered / correct when answered")
    print(f"  {'min_confidence':<14}" + "".join(f"{q:>20}" for q in questions))
    for threshold in THRESHOLDS:
        strict = at_min_confidence(report, questions, cases, threshold)
        cells = "".join(
            f"{f'{s.answered:>2}/{s.total}  {s.correct:>2}/{s.answered:<2}':>20}"
            for s in strict.questions.values()
        )
        print(f"  {threshold:<14}{cells}")


with TypeSafeClient() as client:
    benchmark("TypeSafe (jev-latest)", TypeSafe(client))

benchmark("Laya", Laya(laya.load("convaiinnovations/laya")))
