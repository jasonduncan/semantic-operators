"""Each answer records which model answered and what the call used. Offline."""

import httpx2
import typesafe_sdk as ts

from semantic_operators import Boolean, Call, Choice
from semantic_operators.providers.laya import Laya
from semantic_operators.providers.typesafe import TypeSafe

questions = {"q": Boolean("Is it?"), "pick": Choice("Which?", {"a": None, "b": None})}


def test_typesafe_answers_share_the_reported_call():
    body = {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 12, "output_tokens": 0},
        "answers": {
            "q": {"type": "noul", "noul": 0.9},
            "pick": {"type": "choice", "choice": "a", "confidence": 0.7,
                     "probabilities": {"a": 0.7, "b": 0.3}},
        },
    }
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=body))
    client = ts.TypeSafeClient(api_key="test", transport=transport,
                               retry=ts.RetryPolicy(max_retries=0))
    answers = TypeSafe(client, model="jev-latest").ask("hello", questions)

    assert answers["q"].call == Call("TypeSafe", "jev-1.13.0", 12, 0)
    assert answers["q"].call is answers["pick"].call  # one call, one record


def test_laya_call_keeps_only_what_was_reported():
    class Model:
        def predict(self, state, qs):
            return {"model": "laya-rl-agent", "usage": {"input_tokens": 7},
                    "answers": {"q": {"noul": 0.2},
                                "pick": {"choice": "b", "probabilities": {"a": 0.4, "b": 0.6}}}}

    answers = Laya(Model()).ask("hello", questions)
    assert answers["pick"].call == Call("Laya", "laya-rl-agent", 7, None)


def test_malformed_call_details_become_none_not_errors():
    class Model:
        def predict(self, state, qs):
            return {"model": 3, "usage": {"input_tokens": -1, "output_tokens": True},
                    "answers": {"q": {"noul": 0.8},
                                "pick": {"choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}}}

    answer = Laya(Model()).ask("hello", questions)["q"]
    assert answer.value is True and answer.call == Call("Laya")
