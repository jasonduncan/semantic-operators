"""Ask Jev three questions about one message.

Run:  uv run --env-file .env --extra jev python examples/hello.py
"""

from typesafe_sdk import TypeSafeClient

from semantic_operators import Boolean, Choice, Score
from semantic_operators.providers.jev import Jev

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

# The SDK reads TYPESAFE_API_KEY from the environment; our library never does.
with TypeSafeClient() as client:
    provider = Jev(client)
    answers = provider.ask(message, questions)

print(f"Message: {message}\n")
for name, answer in answers.items():
    probabilities = ", ".join(f"{k}={p:.2f}" for k, p in answer.probabilities.items())
    print(f"{name:>13}: {answer.value!r:<12} ({probabilities})")
