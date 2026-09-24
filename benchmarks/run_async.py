"""What concurrency buys each provider, and both providers benchmarked at once.

Run:  uv run --env-file .env --extra jev --extra laya python benchmarks/run_async.py
Makes 63 Jev API calls (3 runs x (1 warm-up + 20 cases)).
"""

import asyncio
import statistics

import laya
from typesafe_sdk import AsyncTypeSafeClient

from semantic_operators.bench import Report, run_async
from semantic_operators.providers.jev import AsyncJev
from semantic_operators.providers.laya import AsyncLaya
from support_tickets import cases, questions


def show(label: str, report: Report) -> None:
    correct = sum(s.correct for s in report.questions.values())
    total = sum(s.total for s in report.questions.values())
    print(f"  {label:<28} total {report.total_ms:>6.0f} ms   "
          f"per call: median {statistics.median(report.latencies_ms):>4.0f} ms, "
          f"slowest {max(report.latencies_ms):>4.0f} ms   correct {correct}/{total}")


async def main() -> None:
    async with AsyncTypeSafeClient() as client:
        jev = AsyncJev(client)
        lay = AsyncLaya(laya.load("convaiinnovations/laya"))

        for name, provider in [("Jev", jev), ("Laya", lay)]:
            print(f"\n{name}")
            for concurrency in (1, 8):
                report = await run_async(provider, questions, cases, concurrency=concurrency)
                show(f"concurrency {concurrency}", report)

        print("\nBoth at once (concurrency 8 each)")
        jev_report, laya_report = await asyncio.gather(
            run_async(jev, questions, cases, concurrency=8),
            run_async(lay, questions, cases, concurrency=8),
        )
        show("Jev", jev_report)
        show("Laya", laya_report)


asyncio.run(main())
