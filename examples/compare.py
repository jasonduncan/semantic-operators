"""Ask Jev (hosted by TypeSafe) and Laya (local) the same questions, through the same interface.

Run:  uv run --env-file .env --extra typesafe --extra laya python examples/compare.py
The first run downloads the Laya checkpoint (~800 MB) from Hugging Face.
"""

import time

import laya
from typesafe_sdk import TypeSafeClient

from semantic_operators import Boolean, Choice, Provider, Score
from semantic_operators.providers.laya import Laya
from semantic_operators.providers.typesafe import TypeSafe

message = "I was charged twice for my subscription this month and I'm furious."

questions = {
    "is_complaint": Boolean("Is the customer complaining?"),
    "department": Choice(
        "Which team should handle this?",
        {
            "billing": "Charges, invoices, payments, or refunds.",
            "technical": "A product defect or difficulty using the product.",
            "other": "Anything else.",
        },
    ),
    "urgency": Score("How urgent is this request?", ["low", "medium", "high"]),
}


def show(name: str, provider: Provider) -> None:
    start = time.perf_counter()
    answers = provider.ask(message, questions)
    ms = (time.perf_counter() - start) * 1000
    print(f"\n{name} ({ms:.0f} ms)")
    for question, answer in answers.items():
        value = f"{answer.value:.2f}" if isinstance(answer.value, float) else repr(answer.value)
        probabilities = ", ".join(f"{k}={p:.2f}" for k, p in answer.probabilities.items())
        print(f"  {question:>13}: {value:<12} confidence {answer.confidence:.2f}  ({probabilities})")


print(f"Message: {message}")

with TypeSafeClient() as client:
    show("TypeSafe (jev-latest)", TypeSafe(client))

model = laya.load("convaiinnovations/laya")  # English checkpoint, runs on this machine
laya_provider = Laya(model)
laya_provider.ask(message, questions)  # warm-up: the first call pays one-time setup costs
show("Laya", laya_provider)
