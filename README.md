# Semantic Operators

```sh
pip install "semantic-operators[typesafe]"
```

**Write a semantic judgment once, run it on any System One model, and get a "don't
know" instead of a guess when the model isn't sure.**

System One models are small, fast models that classify instead of generating text,
as opposed to a chat LLM. You ask them typed questions (yes/no, pick one, rate on a
scale) about text or data, and they return answers with probabilities.
[Jev](https://typesafe.ai) was the first; [Laya](https://huggingface.co/convaiinnovations/laya)
is an open-weight, Jev-compatible alternative you can run locally. More are coming.

| Provider | Models | Where it runs | Install |
|----------|--------|---------------|---------|
| `providers.typesafe.TypeSafe` | Jev (`jev-latest`, or pin a version) | TypeSafe's hosted API (needs `TYPESAFE_API_KEY`) | `[typesafe]` |
| `providers.laya.Laya` | Laya checkpoints (English, multilingual, typed-decisions) | on your machine (~800 MB download on first use) | `[laya]` |

A provider is the service or runtime you talk to; the model is a setting.

## Install

```sh
pip install "semantic-operators[typesafe]"        # TypeSafe (hosted Jev)
pip install "semantic-operators[laya]"            # Laya (local; pulls in torch)
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
order), `confidence` (the probability of its own answer), `raw` (the provider's
own answer object), and `call` (which model answered, below).

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

## Named operators

This is where the library gets its name. An **operator** is a semantic judgment
defined once, with a name, and used anywhere:

```python
from semantic_operators import Boolean, Choice, Score
from semantic_operators.operators import Operator, apply

is_complaint = Operator("is_complaint", Boolean("Is the customer complaining?"))
urgency = Operator("urgency", Score("How urgent is this?", ["low", "medium", "high"]))

is_complaint(provider, message).value                        # one operator, one call
answers = apply(provider, message, [is_complaint, urgency])  # several, still one call
answers["urgency"].value
```

- **Combine freely.** System One models answer many questions in one pass, so `apply`
  asks any set of operators in a single provider call. Names must be unique.
- **Provider-neutral.** An operator doesn't hold a provider; you pass one in, so the
  same operator runs on TypeSafe, Laya, or anything else.
- **Wording is part of the operator.** It changes the answers (see the benchmark), so
  keep operators in code, under version control, and benchmark them as they are.
  `operators.questions([...])` turns them into the dict `bench.run` takes.
- **Async:** `await op.call_async(provider, state)` and `await apply_async(...)`.

`examples/triage.py` builds ticket triage from three operators, sending anything the
model isn't sure about to a person. Add `--laya` to run the same code locally.

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

## Which model answered

`jev-latest` moves over time, so every answer records what its provider call reported:

```python
answers["department"].call
# Call(provider='TypeSafe', model='jev-1.13.0', input_tokens=291, output_tokens=20)
```

All answers from one call share one `Call`. `model` is exactly what the provider
reported, which can differ from what you asked for (above, `jev-latest`). Laya reports a
fixed agent name (`laya-rl-agent`), not which checkpoint answered. Anything a provider
doesn't report is `None`.

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

Provider output is checked before it becomes an `Answer`: probabilities must be finite,
between 0 and 1, sum to 1 (allowing for the providers' rounding), and agree with the
answer. A malformed response raises `ProviderError` rather than looking like a confident
answer.

Retries and timeouts belong to the client you build. The TypeSafe SDK retries by
default; to have every failure reach you (for example, when something above you does
its own retrying), turn that off:

```python
TypeSafeClient(retry=RetryPolicy(max_retries=0, timeout=10.0))
``` Questions check themselves too: a `Choice` needs at least two distinct options,
a `Score` at least two distinct levels, and a bad definition raises `ValueError`.

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
  types.py          Boolean, Choice, Score, Answer, Call, make_answer: our vocabulary
  provider.py       Provider and AsyncProvider (one method each)
  errors.py         ProviderError, the one error every provider raises
  providers/typesafe.py  translates to/from the TypeSafe SDK
  providers/laya.py translates to/from the laya package
  operators.py      (higher layer) named operators: define once, combine in one call
  bench.py          (higher layer) run labeled cases through a provider, score them
examples/
  hello.py          one real call to Jev
  compare.py        the same questions through TypeSafe and Laya
  triage.py         ticket triage built from named operators
benchmarks/
  support_tickets.py  20 labeled messages + the questions
  run.py              runs the suite through TypeSafe and Laya
  run_async.py        concurrency, and both providers at once
  run_confidence.py   the "don't know" trade-off at several min_confidence levels
tests/                offline tests (uv run --extra typesafe pytest); CI runs them
ROADMAP.md            where this is headed
```

## Layers

Semantic Operators is built in layers inside one package:

1. **Base layer:** a clean, provider-neutral abstraction over System One
   models: `types.py`, `provider.py`, `errors.py`, `providers/`.
2. **Higher layers:** built only on the base layer: `operators.py` (named operators)
   and `bench.py` (benchmarking).

The base layer never imports from a higher layer, so it could later be split out as its
own package without changing how it's used.

## Rules

- The library never reads API keys or environment variables. You build the client.
- The core has no dependencies. Each provider's SDK is an optional extra (`[typesafe]`, `[laya]`).
- Our names, not the provider's: `Boolean`, not `noul`.

## Not here yet (on purpose)

Operators are just named questions for now. **Combining** them is where this is
headed: conditions ("ask B only when A says yes"), chains, and small decision flows
built from operators. Also planned: a provider-neutral call timeout, and suites as data
files. See [ROADMAP.md](ROADMAP.md).

## License

MIT
