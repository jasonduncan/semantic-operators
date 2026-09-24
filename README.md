# Semantic Operators

One small interface for **System One models**: fast models that answer structured
questions about text or data with probabilities, not prose. [Jev](https://typesafe.ai)
was the first; [Laya](https://huggingface.co/convaiinnovations/laya) is an open-weight,
Jev-compatible alternative you can run locally. More are coming. This library lets you
write your code once and swap the model underneath.

| Provider | Where it runs | Install |
|----------|---------------|---------|
| `providers.jev.Jev`   | TypeSafe's hosted API (needs `TYPESAFE_API_KEY`) | `[jev]`  |
| `providers.laya.Laya` | on your machine (~800 MB download on first use)  | `[laya]` |

## The whole idea

A System One model is asked **named questions about a piece of state** and returns an
answer with probabilities for each. There are three kinds of question:

| Question  | You give it                                   | `answer.value`                       |
|-----------|-----------------------------------------------|--------------------------------------|
| `Boolean` | instructions (+ optional true/false meanings) | `True` / `False`                     |
| `Choice`  | instructions + named options                  | the chosen option name               |
| `Score`   | instructions + ordered rubric levels          | expected level as a float, e.g. `1.7` |

Every `Answer` also carries `probabilities` (a dict, in the question's option/level
order) and `raw` (the provider's own answer object).

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

Swapping to Laya changes only how the provider is built:

```python
import laya
from semantic_operators.providers.laya import Laya

provider = Laya(laya.load("convaiinnovations/laya"))   # or Laya(laya.Router())
answers = provider.ask(state, questions)                # same questions, same Answer type
```

Compare both side by side:

```sh
uv run --env-file .env --extra jev --extra laya python examples/compare.py
```

## Layout

```
src/semantic_operators/
  types.py          Boolean, Choice, Score, Answer: our vocabulary
  provider.py       the Provider interface (one method)
  providers/jev.py  translates to/from the TypeSafe SDK
  providers/laya.py translates to/from the laya package
examples/hello.py   one real call to Jev
examples/compare.py the same questions through Jev and Laya
```

## Layers

Semantic Operators is built in layers inside one package:

1. **Base layer (today):** a clean, provider-neutral abstraction over System One
   models: `types.py`, `provider.py`, `providers/`.
2. **Higher layers (later):** reusable named operators, composition, benchmarking.
   These are built only on the base layer.

The base layer never imports from a higher layer, so it could later be split out as its
own package without changing how it's used.

## Rules

- The library never reads API keys or environment variables. You build the client.
- The core has no dependencies. Each provider's SDK is an optional extra (`[jev]`, `[laya]`).
- Our names, not the provider's: `Boolean`, not `noul`.

## Not here yet (on purpose)

Async, benchmarking, reusable named operators, error types, and
"don't know" answers. Each will be added as its own small step.
