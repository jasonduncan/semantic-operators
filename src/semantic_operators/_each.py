"""One question about each of many states: one call per state, results in input order.

Shared by ``rerank`` and ``filtering``. Each state gets its own call, so an item's answer
can't depend on the other items. A ``ProviderError`` is recorded for its own item and
the rest carry on; any other exception propagates (and, async, cancels the rest).
"""

import asyncio
from collections.abc import Sequence

from .errors import ProviderError
from .provider import AsyncProvider, Provider
from .types import Answer, Question, State

Result = Answer | ProviderError

_NAME = "q"


def ask_each(provider: Provider, question: Question, states: Sequence[State]) -> list[Result]:
    results: list[Result] = []
    for state in states:
        try:
            results.append(provider.ask(state, {_NAME: question})[_NAME])
        except ProviderError as error:
            results.append(error)
    return results


async def ask_each_async(provider: AsyncProvider, question: Question, states: Sequence[State],
                         concurrency: int) -> list[Result]:
    check_concurrency(concurrency)
    slots = asyncio.Semaphore(concurrency)

    async def one(state: State) -> Result:
        async with slots:
            try:
                return (await provider.ask(state, {_NAME: question}))[_NAME]
            except ProviderError as error:
                return error

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(state)) for state in states]
    return [task.result() for task in tasks]


def check_concurrency(concurrency: int) -> None:
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
