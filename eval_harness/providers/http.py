"""
Day 19 (Phase 30): a tiny HTTP transport abstraction -- the same
optional-dependency pattern `providers/llm.py`/`providers/embedding.py`
already established for OpenAI, applied here so the Project 1 integration
adapter (`adapters/production_rag.py`) can be fully unit-tested with
`FakeHttpClient` (no live server, no network) and used for real via
`HttpxHttpClient` without this package taking on a hard dependency on any
particular HTTP library.

Why a Protocol around "POST and get back (status, json)" rather than just
importing `httpx` directly in the adapter: the adapter's whole job is
talking to an already-running, already-deployed system over HTTP -- that's
the one place in this harness an external network call is actually the
point, not a judge-LLM side call. Keeping the transport itself swappable
means the adapter's response-mapping logic (the part actually worth
testing) is testable without ever opening a socket.
"""

from typing import Any, Protocol, runtime_checkable


class HttpError(Exception):
    """A transport-level failure -- connection refused, DNS failure, a
    timeout, a malformed response body that couldn't be parsed as JSON at
    all. NOT raised for an ordinary HTTP error status (4xx/5xx); those come
    back as a normal `(status, body)` return so callers can inspect
    whatever error detail the server put in the body, same as a successful
    response."""


@runtime_checkable
class HttpClient(Protocol):
    async def post(
        self, url: str, *, json: dict, headers: dict, timeout_s: float = 30.0
    ) -> tuple[int, dict]:
        """POST `json` to `url` with `headers`. Returns `(status_code,
        parsed_json_body)` for ANY response the server actually sent, 2xx
        or not. Raises `HttpError` only when no response was obtained at
        all (the transport itself failed) or the body couldn't be parsed as
        JSON."""
        ...


class FakeHttpClient:
    """Scripted HTTP responses -- the same two modes `providers/llm.py`'s
    `FakeJsonLLM` already established: a `default` response returned for
    every call, or a `queue` of per-call responses consumed in order (for a
    test that needs call N to differ from call N+1, e.g. a transient
    failure followed by a success). Every call is recorded in `.calls` so
    a test can assert on exactly what the adapter sent -- the URL, the
    request body, the headers (useful for checking the API key actually
    went out on the expected header)."""

    def __init__(
        self,
        *,
        default: tuple[int, dict] | None = None,
        queue: list[tuple[int, dict]] | None = None,
        raises: Exception | None = None,
    ):
        self._default = default
        self._queue = list(queue) if queue is not None else None
        self._raises = raises
        self.calls: list[dict] = []

    async def post(self, url: str, *, json: dict, headers: dict, timeout_s: float = 30.0) -> tuple[int, dict]:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout_s": timeout_s})
        if self._raises is not None:
            raise self._raises
        if self._queue is not None:
            if not self._queue:
                raise HttpError("FakeHttpClient's response queue is exhausted.")
            return self._queue.pop(0)
        if self._default is not None:
            return self._default
        raise HttpError("FakeHttpClient was given neither `default` nor `queue` -- nothing to return.")


class HttpxHttpClient:
    """The real HTTP transport, via `httpx` (optional dependency, lazy
    import -- same discipline as `OpenAIJsonLLM`/`OpenAIEmbeddingProvider`:
    importing this module never requires `httpx` installed, only
    constructing this class does). Install with `pip install -r
    requirements-httpx.txt`.

    `_transport` is an escape hatch, not a normal constructor argument:
    unlike an OpenAI provider, `httpx` ships `httpx.MockTransport`, which
    lets this class's actual request/response handling (status codes,
    error wrapping, JSON parsing) be unit-tested for real -- no live
    server, no network, no API key -- rather than excluded from coverage
    the way the OpenAI providers' live call paths are. Leave it unset for
    real use; a real `httpx.AsyncClient` picks its own transport."""

    def __init__(self, *, _transport: object | None = None) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "HttpxHttpClient needs the `httpx` package. Install it with: "
                "pip install -r requirements-httpx.txt"
            ) from exc
        self._httpx = httpx
        self._transport = _transport

    async def post(self, url: str, *, json: dict, headers: dict, timeout_s: float = 30.0) -> tuple[int, Any]:
        try:
            async with self._httpx.AsyncClient(timeout=timeout_s, transport=self._transport) as client:
                resp = await client.post(url, json=json, headers=headers)
        except self._httpx.HTTPError as exc:
            raise HttpError(f"request to {url} failed: {type(exc).__name__}: {exc}") from exc
        try:
            body = resp.json()
        except ValueError as exc:
            raise HttpError(f"response from {url} was not valid JSON: {exc}") from exc
        return resp.status_code, body
