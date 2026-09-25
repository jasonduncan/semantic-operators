"""The one error type every provider raises.

Whatever went wrong underneath (network, bad API key, rate limit, a model that
failed or returned something unexpected), you catch ``ProviderError``. The
original exception is kept as ``__cause__`` (shown in tracebacks) when you need
the details.
"""


class ProviderError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
