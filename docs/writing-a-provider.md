# Writing a provider

A provider is anything with one method:

```python
def ask(self, state, questions: Mapping[str, Question]) -> dict[str, Answer]: ...
```

It's a `Protocol`, so there's nothing to inherit from or register. A new System One
model (a Jev clone, a hosted service, a classifier you trained) needs one provider, and
then every operator, flow, reranker, benchmark, and the `semop` CLI and MCP server work
with it unchanged. `providers/typesafe.py` and `providers/laya.py` are about 100 lines
each and are the best references.

## What a provider does

1. **Translate our questions into the model's API.** `Boolean`, `Choice`, and `Score`
   are our vocabulary; the model may call them something else (TypeSafe says `noul`).
2. **Make one call** for all the questions, if the model can answer several at once.
3. **Translate the answers back**, building each one with `make_answer`.
4. **Raise `ProviderError` for anything that goes wrong**, never the SDK's own exceptions.

The provider doesn't read API keys or environment variables: whoever uses it builds the
client (or loads the model) and passes it in.

## A skeleton

```python
from collections.abc import Mapping
from typing import Any

from semantic_operators import (Answer, Boolean, Call, Choice, ProviderError, Question,
                                Score, State, make_answer)
from semantic_operators.types import complement, token_count


class MyModel:
    def __init__(self, client: Any, model: str = "my-model-latest") -> None:
        self.client, self.model = client, model

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        request = {name: _to_api(q) for name, q in questions.items()}
        try:
            response = self.client.classify(state, request, model=self.model)
        except MyModelError as error:  # the SDK's own base error class
            raise ProviderError("MyModel", str(error)) from error
        try:
            call = Call("MyModel", response.get("model"),
                        token_count(response.get("input_tokens")), None)
            return {name: _from_api(q, response["answers"][name], call)
                    for name, q in questions.items()}
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderError("MyModel", f"unexpected response: {error!r}") from error


def _to_api(question: Question) -> dict[str, Any]:
    match question:
        case Boolean():
            return {"kind": "yes_no", "question": question.instructions}
        case Choice():
            return {"kind": "pick", "question": question.instructions,
                    "labels": dict(question.options)}
        case Score():
            return {"kind": "rate", "question": question.instructions,
                    "scale": list(question.levels)}


def _from_api(question: Question, answer: dict[str, Any], call: Call) -> Answer:
    match question:
        case Boolean():
            p = answer["p_yes"]
            return make_answer(question, p > 0.5, {"true": p, "false": complement(p)},
                               answer, call=call)
        case Choice():
            probabilities = {o: answer["probabilities"][o] for o in question.options}
            return make_answer(question, answer["label"], probabilities, answer, call=call)
        case Score():
            probabilities = {level: answer["probabilities"][i]
                             for i, level in enumerate(question.levels)}
            return make_answer(question, answer["expected"], probabilities, answer, call=call)
```

## The rules `make_answer` enforces

Build every answer with `make_answer(question, value, probabilities, raw, call=call)`.
It gives every provider the same behavior for ties and `min_confidence`, and it checks
the model's output. Anything malformed raises `ValueError`; turn that into
`ProviderError`, as the skeleton does, so a broken response never looks like a
confident answer.

- **Probabilities** cover exactly the question's outcomes: `"true"`/`"false"`, the option
  names, or the level texts. They're finite, between 0 and 1, and sum to 1, allowing
  for rounding to 2 decimals.
- **Keep the question's order:** build Choice and Score probabilities by iterating over
  `question.options` / `question.levels`, not over the model's response.
- **The value agrees with them.** A Boolean's value is the more likely side; a Choice's
  is the most likely option; a Score's value is the expected level index
  (`sum(i * p_i)`), which may fall between levels.
- **Booleans:** if the model reports only p(true), use `complement(p)` for p(false), not
  `1 - p`. Plain subtraction adds float noise (`1 - 0.07` is `0.9299999999999999`) that
  can push a correct answer under its `min_confidence`.
- **`raw`** is the model's own answer object, kept for callers who need more.
- **`call`** is what the model reported about the call: `Call(provider, model,
  input_tokens, output_tokens)`, one shared by every answer from that call. Report the
  model it says actually answered, and leave anything unreported as `None`, not zero.

## Async, errors, and timeouts

- **Async twin:** add an `AsyncMyModel` with `async def ask`, sharing the translation
  functions. A local model with no async API can run its sync `ask` in
  `asyncio.to_thread`, holding a `threading.Lock` around the prediction itself (see
  `AsyncLaya`), so a cancelled call can't let a second prediction overlap the first.
- **Timeouts:** if the SDK times out, raise `ProviderTimeout` (a `ProviderError` and a
  `TimeoutError`). Callers add their own limits with `with_timeout`; you don't need to.
- **Retries** belong to the client the caller builds, not the provider.

## Testing it

Test offline by giving the provider a fake client (or, for an HTTP SDK, a fake
transport) that returns canned responses. See `tests/test_errors.py` for the TypeSafe
SDK with a mocked HTTP layer. Check at least: one call per `ask`, probabilities in
question order, a malformed response and an SDK error each raising `ProviderError`,
and one real call. Then run `benchmarks/run.py` with it to see how it compares.
