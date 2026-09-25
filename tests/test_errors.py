"""Every provider failure surfaces as ProviderError. Offline: no network, no model weights."""

import httpx2
import pytest
import typesafe_sdk as ts

from semantic_operators import Boolean, ProviderError
from semantic_operators.providers.laya import Laya
from semantic_operators.providers.typesafe import TypeSafe

questions = {"q": Boolean("Is it?")}


def typesafe_returning(status: int, body: dict) -> TypeSafe:
    # The real SDK, with its HTTP layer replaced by a canned response.
    transport = httpx2.MockTransport(lambda request: httpx2.Response(status, json=body))
    client = ts.TypeSafeClient(api_key="test", transport=transport,
                               retry=ts.RetryPolicy(max_retries=0))
    return TypeSafe(client)


def test_typesafe_api_error_becomes_provider_error():
    provider = typesafe_returning(401, {"error": {"message": "bad key"}})
    with pytest.raises(ProviderError) as caught:
        provider.ask("hello", questions)
    assert caught.value.provider == "TypeSafe"
    assert isinstance(caught.value.__cause__, ts.TypeSafeError)


def test_typesafe_missing_answer_becomes_provider_error():
    provider = typesafe_returning(200, {"model": "jev", "usage": {}, "answers": {}})
    with pytest.raises(ProviderError, match="unexpected response"):
        provider.ask("hello", questions)


def test_laya_failure_becomes_provider_error():
    class BrokenModel:
        def predict(self, state, questions):
            raise ValueError("options do not fit")

    with pytest.raises(ProviderError, match="options do not fit") as caught:
        Laya(BrokenModel()).ask("hello", questions)
    assert isinstance(caught.value.__cause__, ValueError)
