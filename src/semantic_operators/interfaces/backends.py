"""Building a provider from command-line settings, for the CLI and the MCP server.

This is application code, so unlike the library it may read the environment: the
TypeSafe SDK reads ``TYPESAFE_API_KEY`` itself. The provider and model are fixed when
the process starts. Nothing in a request can change them, so an agent calling the MCP
tool can't switch a local-only server to a paid hosted API.
"""

import asyncio
import contextlib
import os
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from ..provider import AsyncProvider, with_timeout
from ..types import Answer, Question, State

PROVIDERS = ("typesafe", "laya")
DEFAULT_PROVIDER = "typesafe"
DEFAULT_MODELS = {"typesafe": "jev-latest", "laya": "convaiinnovations/laya"}


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str | None = None
    timeout: float | None = None

    @property
    def model_name(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]


@contextlib.asynccontextmanager
async def open_provider(settings: Settings) -> AsyncIterator[AsyncProvider]:
    """A ready provider for ``settings``, closed (for TypeSafe) when the block ends.

    Laya is loaded lazily, on the first question, so a server starts and lists its
    tools right away. It's loaded once and reused for every call after that.
    """
    if settings.provider == "typesafe":
        from typesafe_sdk import AsyncTypeSafeClient

        from ..providers.typesafe import AsyncTypeSafe

        async with AsyncTypeSafeClient() as client:
            yield _limit(AsyncTypeSafe(client, model=settings.model_name), settings)
    elif settings.provider == "laya":
        yield _limit(_LazyLaya(settings.model_name), settings)
    else:
        raise ValueError(f"unknown provider {settings.provider!r}; choose from {PROVIDERS}")


def missing_setup(settings: Settings) -> str | None:
    """What's missing before the provider can run (its package, or TypeSafe's API key),
    as a message for the user, or ``None`` if everything's in place."""
    return missing_dependency(settings) or missing_key(settings)


def missing_key(settings: Settings) -> str | None:
    if settings.provider == "typesafe" and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        return ("TypeSafe needs an API key: set TYPESAFE_API_KEY, "
                "or use --provider laya to run a local model instead")
    return None


def missing_dependency(settings: Settings) -> str | None:
    """How to install the provider's package if it's missing, else ``None``."""
    module = {"typesafe": "typesafe_sdk", "laya": "laya"}[settings.provider]
    try:
        __import__(module)
    except ImportError:
        return (f"the {settings.provider} provider isn't installed; install it with: "
                f'pip install "semantic-operators[{settings.provider}]"')
    return None


def _limit(provider: AsyncProvider, settings: Settings) -> AsyncProvider:
    return provider if settings.timeout is None else with_timeout(provider, settings.timeout)


class _LazyLaya:
    """``AsyncLaya``, loading the model on first use. Concurrent first calls load it once."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._provider: AsyncProvider | None = None
        self._loading = asyncio.Lock()

    async def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        async with self._loading:
            if self._provider is None:
                from ..providers.laya import AsyncLaya

                model = await asyncio.to_thread(_load_laya, self.model_name)
                self._provider = AsyncLaya(model)
        return await self._provider.ask(state, questions)


def _load_laya(model_name: str) -> Any:
    import laya

    # stdout is reserved for results (and MCP messages): keep any loader output off it.
    with contextlib.redirect_stdout(sys.stderr):
        return laya.load(model_name)
