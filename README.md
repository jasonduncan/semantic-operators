# Semantic Operators

One small interface for **System One models**: fast models that answer structured
questions about text or data with probabilities, not prose. [Jev](https://typesafe.ai)
is the first. Clones and competitors are coming, and this library lets you write your
code once and swap the model underneath.

## The whole idea

A System One model is asked **named questions about a piece of state** and returns an
answer with probabilities for each. There are three kinds of question:

| Question  | You give it                                   | `answer.value`                       |
|-----------|-----------------------------------------------|--------------------------------------|
| `Boolean` | instructions (+ optional true/false meanings) | `True` / `False`                     |
| `Choice`  | instructions + named options                  | the chosen option name               |
| `Score`   | instructions + ordered rubric levels          | expected level as a float, e.g. `1.7` |

Every `Answer` also carries `probabilities` (a dict) and `raw` (the provider's own
answer object).

A **provider** is anything with one method:

```python
def ask(self, state, questions: dict[str, Question]) -> dict[str, Answer]
```

That's the entire abstraction.

## Quick start

```sh
echo "TYPESAFE_API_KEY=..." > .env
uv run --env-file .env --extra jev python examples/hello.py
```

```python
from typesafe_sdk import TypeSafeClient
from semantic_operators import Boolean, Choice, Score
from semantic_operators.providers.jev import Jev

with TypeSafeClient() as client:          # you create and own the SDK client
    jev = Jev(client)                     # model defaults to "jev-latest"
    answers = jev.ask(
        "I was charged twice and I'm furious.",
        {
            "is_complaint": Boolean("Is the customer complaining?"),
            "department": Choice("Which team should handle this?",
                                 {"billing": "Payments, refunds", "other": "Anything else"}),
            "urgency": Score("How urgent is this?", ["low", "medium", "high"]),
        },
    )

answers["department"].value           # "billing"
answers["department"].probabilities   # {"billing": 0.97, "other": 0.03}
```

## Layout

```
src/semantic_operators/
  types.py          Boolean, Choice, Score, Answer: our vocabulary
  provider.py       the Provider interface (one method)
  providers/jev.py  translates to/from the TypeSafe SDK
examples/hello.py   one real call to Jev
```

## Rules

- The library never reads API keys or environment variables. You build the client.
- The core has no dependencies. Each provider's SDK is an optional extra (`[jev]`).
- Our names, not the provider's: `Boolean`, not `noul`.

## Not here yet (on purpose)

Async, a second provider, benchmarking, reusable named operators, error types, and
"don't know" answers. Each will be added as its own small step.
