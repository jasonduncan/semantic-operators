"""The provider interfaces: anything with this ``ask`` method is a provider.

They're Protocols, so providers don't inherit from anything or register anywhere.
``AsyncProvider`` is the same contract with ``ask`` awaited.
"""

from collections.abc import Mapping
from typing import Protocol

from .types import Answer, Question, State


class Provider(Protocol):
    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        """Answer every named question about ``state``. Returns answers by the same names."""
        ...


class AsyncProvider(Protocol):
    async def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        """Answer every named question about ``state``. Returns answers by the same names."""
        ...
