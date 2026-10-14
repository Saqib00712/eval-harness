import pytest

from eval_harness.providers.http import FakeHttpClient, HttpError


async def test_fake_http_client_returns_default_response():
    client = FakeHttpClient(default=(200, {"answer": "hi"}))
    status, body = await client.post("http://x/ask", json={"q": "a"}, headers={"H": "1"})
    assert status == 200
    assert body == {"answer": "hi"}


async def test_fake_http_client_records_calls():
    client = FakeHttpClient(default=(200, {}))
    await client.post("http://x/ask", json={"q": "a"}, headers={"X-API-Key": "k"})
    assert client.calls == [
        {"url": "http://x/ask", "json": {"q": "a"}, "headers": {"X-API-Key": "k"}, "timeout_s": 30.0}
    ]


async def test_fake_http_client_queue_consumed_in_order():
    client = FakeHttpClient(queue=[(500, {"error": "boom"}), (200, {"answer": "ok"})])
    first = await client.post("http://x/ask", json={}, headers={})
    second = await client.post("http://x/ask", json={}, headers={})
    assert first == (500, {"error": "boom"})
    assert second == (200, {"answer": "ok"})


async def test_fake_http_client_queue_exhausted_raises():
    client = FakeHttpClient(queue=[(200, {})])
    await client.post("http://x/ask", json={}, headers={})
    with pytest.raises(HttpError, match="exhausted"):
        await client.post("http://x/ask", json={}, headers={})


async def test_fake_http_client_no_response_configured_raises():
    client = FakeHttpClient()
    with pytest.raises(HttpError, match="neither"):
        await client.post("http://x/ask", json={}, headers={})


async def test_fake_http_client_raises_configured_exception():
    client = FakeHttpClient(raises=HttpError("connection refused"))
    with pytest.raises(HttpError, match="connection refused"):
        await client.post("http://x/ask", json={}, headers={})


def test_httpx_http_client_requires_httpx_installed():
    pytest.importorskip("httpx")
    from eval_harness.providers.http import HttpxHttpClient

    # construction-only -- no real network call, no live server needed
    client = HttpxHttpClient()
    assert client is not None


# ---- HttpxHttpClient.post, via httpx.MockTransport -- real request/response
# handling exercised with no live server and no network, using the same
# library that handles a real call in production ----

def _mock_client(handler):
    httpx = pytest.importorskip("httpx")
    from eval_harness.providers.http import HttpxHttpClient

    return HttpxHttpClient(_transport=httpx.MockTransport(handler))


async def test_httpx_client_returns_status_and_parsed_json():
    httpx = pytest.importorskip("httpx")

    def handler(request):
        return httpx.Response(200, json={"answer": "hi"})

    client = _mock_client(handler)
    status, body = await client.post("http://x/ask", json={"q": "a"}, headers={})
    assert status == 200
    assert body == {"answer": "hi"}


async def test_httpx_client_passes_through_error_status_without_raising():
    httpx = pytest.importorskip("httpx")

    def handler(request):
        return httpx.Response(401, json={"detail": "invalid API key"})

    client = _mock_client(handler)
    status, body = await client.post("http://x/ask", json={}, headers={})
    assert status == 401
    assert body == {"detail": "invalid API key"}


async def test_httpx_client_sends_json_body_and_headers():
    httpx = pytest.importorskip("httpx")
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read()
        return httpx.Response(200, json={})

    client = _mock_client(handler)
    await client.post("http://x/ask", json={"question": "hi"}, headers={"X-API-Key": "k"})
    assert seen["headers"]["x-api-key"] == "k"
    assert b"hi" in seen["body"]


async def test_httpx_client_non_json_response_raises_http_error():
    httpx = pytest.importorskip("httpx")
    from eval_harness.providers.http import HttpError

    def handler(request):
        return httpx.Response(200, text="not json")

    client = _mock_client(handler)
    with pytest.raises(HttpError, match="not valid JSON"):
        await client.post("http://x/ask", json={}, headers={})


async def test_httpx_client_transport_failure_raises_http_error():
    httpx = pytest.importorskip("httpx")
    from eval_harness.providers.http import HttpError

    def handler(request):
        raise httpx.ConnectError("connection refused")

    client = _mock_client(handler)
    with pytest.raises(HttpError, match="request to"):
        await client.post("http://x/ask", json={}, headers={})
