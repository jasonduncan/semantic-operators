# Semantic Operators

One small interface for **System One models**: fast models that answer structured
questions about text or data with probabilities, not prose. [Jev](https://typesafe.ai)
was the first; [Laya](https://huggingface.co/convaiinnovations/laya) is an open-weight,
Jev-compatible alternative you can run locally. More are coming. This library lets you
write your code once and swap the model underneath.

| Provider | Models | Where it runs | Install |
|----------|--------|---------------|---------|
| `providers.typesafe.TypeSafe` | Jev (`jev-latest`, or pin a version) | TypeSafe's hosted API (needs `TYPESAFE_API_KEY`) | `[typesafe]` |
| `providers.laya.Laya` | Laya checkpoints (English, multilingual, typed-decisions) | on your machine (~800 MB download on first use) | `[laya]` |

A provider is the service or runtime you talk to; the model is a setting.

## Install

```sh
pip install "semantic-operators[typesafe]"   # TypeSafe (hosted Jev)
pip install "semantic-operators[laya]"       # Laya (local; pulls in torch)
pip install "semantic-operators[typesafe,laya]"   # both
```

The core alone (`pip install semantic-operators`) has no dependencies.

## The whole idea

A System One model is asked **named questions about a piece of state** and returns an
answer with probabilities for each. There are three kinds of question:

| Question  | You give it                                   | `answer.value`                       |
|-----------|-----------------------------------------------|--------------------------------------|
| `Boolean` | instructions (+ optional true/false meanings) | `True` / `False`                     |
| `Choice`  | instructions + named options                  | the chosen option name               |
| `Score`   | instructions + ordered rubric levels          | expected level as a float, e.g. `1.7` |

Every `Answer` also carries `probabilities` (a dict, in the question's option/level
order), `confidence` (the probability of its own answer), and `raw` (the provider's
own answer object).

A **provider** is anything with one method:

```python
def ask(self, state, questions: dict[str, Question]) -> dict[str, Answer]
```

That's the entire abstraction.

## Quick start

```sh
echo "TYPESAFE_API_KEY=..." > .env
uv run --env-file .env --extra typesafe python examples/hello.py
```

```python
from typesafe_sdk import TypeSafeClient
from semantic_operators import Boolean, Choice, Score
from semantic_operators.providers.typesafe import TypeSafe

with TypeSafeClient() as client:          # you create and own the SDK client
    provider = TypeSafe(client)           # model defaults to "jev-latest"
    answers = provider.ask(
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
uv run --env-file .env --extra typesafe --extra laya python examples/compare.py
```

## "Don't know" answers

A model that's split, or not sure enough, should say so rather than guess. Give any
question a `min_confidence`; below it, the answer comes back **undecided**
(`value is None`, `decided is False`), with its probabilities kept:

```python
department = Choice("Which team?", {"billing": None, "technical": None},
                    min_confidence=0.8)
answer = provider.ask(message, {"department": department})["department"]

if answer.decided:
    route(answer.value)
else:
    send_to_a_human(answer.probabilities)   # still shows what it was leaning toward
```

An exact tie (a Boolean at 0.5, two options equally likely) is always undecided.
Confidence is the model's own view, not a guarantee. `benchmarks/run_confidence.py`
checks whether it means anything: on the support suite, TypeSafe's urgency answers at
`min_confidence=0.8` were right 10 of 10 times (answering half the cases), while
Laya's were right 4 of 7.

## Errors

Every provider raises one error type, whatever went wrong underneath (network, bad
key, rate limit, a model failure, an unexpected response):

```python
from semantic_operators import ProviderError

try:
    answers = provider.ask(state, questions)
except ProviderError as error:
    error.provider     # "TypeSafe" or "Laya"
    error.__cause__    # the original exception, for details
```

## Benchmark

`bench.run(provider, questions, cases)` asks each labeled case all questions in one call
and reports, per question, **accuracy** (Score values are rounded to the nearest level)
and **p(correct)**, the average probability the provider gave the right answer, plus
latency and every miss. Undecided answers are counted separately (`answered`), not as
misses. `bench.at_min_confidence(report, questions, cases, 0.8)` re-scores a run at a
stricter threshold without asking the provider again.

```sh
uv run --env-file .env --extra typesafe --extra laya python benchmarks/run.py
```

`benchmarks/support_tickets.py` holds 20 hand-written, hand-labeled support messages
and the same 3 questions in three wordings. `bench.stability(reports)` reports how often
a provider's decision stays the same when only the wording changes (labels play no part).
It's a smoke test, not a verdict: small, authored, one person's labels.

## Async

Every provider has an async twin with the same contract, `await provider.ask(...)`:

```python
from typesafe_sdk import AsyncTypeSafeClient
from semantic_operators.providers.typesafe import AsyncTypeSafe
from semantic_operators.providers.laya import AsyncLaya

async with AsyncTypeSafeClient() as client:
    answers = await AsyncTypeSafe(client).ask(state, questions)
```

`AsyncLaya` runs the local model in a worker thread, one call at a time. Concurrency
speeds up a hosted API (many requests in flight), not a single local model.
`bench.run_async(provider, questions, cases, concurrency=8)` benchmarks async providers:

```sh
uv run --env-file .env --extra typesafe --extra laya python benchmarks/run_async.py
```

## Layout

```
src/semantic_operators/
  types.py          Boolean, Choice, Score, Answer, make_answer: our vocabulary
  provider.py       Provider and AsyncProvider (one method each)
  errors.py         ProviderError, the one error every provider raises
  providers/typesafe.py  translates to/from the TypeSafe SDK
  providers/laya.py translates to/from the laya package
  bench.py          (higher layer) run labeled cases through a provider, score them
examples/
  hello.py          one real call to Jev
  compare.py        the same questions through TypeSafe and Laya
benchmarks/
  support_tickets.py  20 labeled messages + the questions
  run.py              runs the suite through TypeSafe and Laya
  run_async.py        concurrency, and both providers at once
  run_confidence.py   the "don't know" trade-off at several min_confidence levels
tests/                offline tests (uv run --extra typesafe --extra laya pytest)
```

## Layers

Semantic Operators is built in layers inside one package:

1. **Base layer:** a clean, provider-neutral abstraction over System One
   models: `types.py`, `provider.py`, `errors.py`, `providers/`.
2. **Higher layers:** built only on the base layer. So far: `bench.py`. Later: reusable
   named operators and composition.

The base layer never imports from a higher layer, so it could later be split out as its
own package without changing how it's used.

## Rules

- The library never reads API keys or environment variables. You build the client.
- The core has no dependencies. Each provider's SDK is an optional extra (`[typesafe]`, `[laya]`).
- Our names, not the provider's: `Boolean`, not `noul`.

## Not here yet (on purpose)

Reusable named operators, and batching many inputs into one call (Laya's
`predict_batch`). Each will be added as its own small step.

## License

MIT
