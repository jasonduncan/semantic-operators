"""The provider interfaces: anything with this ``ask`` method is a provider.

They're Protocols, so providers don't inherit from anything or register anywhere.
``AsyncProvider`` is the same contract with ``ask`` awaited.

When something goes wrong, a provider raises ``ProviderError`` (see errors.py),
never its SDK's own exceptions. Answers are built with ``types.make_answer``.

``with_timeout`` gives any async provider a time limit per call.
"""

import asyncio
import math
from collections.abc import Mapping
from typing import Protocol

from .errors import ProviderTimeout
from .types import Answer, Question, State


class Provider(Protocol):
    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        """Answer every named question about ``state``. Returns answers by the same names."""
        ...


class AsyncProvider(Protocol):
    async def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        """Answer every named question about ``state``. Returns answers by the same names."""
        ...


def with_timeout(provider: AsyncProvider, seconds: float) -> AsyncProvider:
    """``provider``, with each ``ask`` limited to ``seconds``.

    A call that runs over raises ``ProviderTimeout``. It's still a provider, so it
    works anywhere one does: operators, ``apply_async``, ``rerank_async`` (where a
    timed-out candidate is simply unscored), benchmarks.

    Stopping to wait isn't always stopping the work. A hosted call is cancelled
    locally, but the server may still finish it (and bill it). A local model keeps
    running its current prediction in the background, and calls wait their turn,
    so the time limit covers that wait too.

    Sync providers have no neutral timeout: set it on the client you build (for
    example ``TypeSafeClient(timeout=5)``); a local model can't be interrupted.
    """
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) \
            or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"seconds must be a positive number, got {seconds!r}")
    return _TimeLimited(provider, seconds)


class _TimeLimited:
    def __init__(self, provider: AsyncProvider, seconds: float) -> None:
        self.provider = provider
        self.seconds = seconds

    async def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        limit = asyncio.timeout(self.seconds)
        try:
            async with limit:
                return await self.provider.ask(state, questions)
        except TimeoutError as error:
            if not limit.expired():
                raise  # the provider's own timeout (already a ProviderTimeout): pass it on
            # Named like the provider's own errors: AsyncTypeSafe reports as "TypeSafe".
            name = type(self.provider).__name__.removeprefix("Async")
            raise ProviderTimeout(name, f"no answer within {self.seconds}s") from error
