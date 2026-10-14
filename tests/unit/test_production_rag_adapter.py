"""
Phase 38: tests for `ProductionRagAdapter` against production-rag's REAL
`AskRequest`/`AskResponse` shape (read directly from
`production-rag/app/models/ask.py` -- see the adapter's own module
docstring for the full before/after). All offline: `FakeHttpClient`
stands in for a live deployment, zero network, zero cost.
"""

import pytest

from eval_harness.adapters.production_rag import ProductionRagAdapter, ProductionRagAdapterError
from eval_harness.models import TestCase
from eval_harness.providers.http import FakeHttpClient, HttpError

TC = TestCase(test_id="t1", input="What is the refund policy?")


def _adapter(http_client, **kwargs):
    return ProductionRagAdapter(http_client, "http://localhost:8000", "test-api-key", **kwargs)


def _citation(**overrides) -> dict:
    base = {
        "source_id": "S1", "chunk_id": "c-001", "document_id": "d-001",
        "page_number": 4, "snippet": "Refunds within 30 days of purchase.", "rerank_score": 0.92,
    }
    base.update(overrides)
    return base


# ---- identity ----

async def test_name_and_version_default():
    adapter = _adapter(FakeHttpClient(default=(200, {})))
    assert adapter.name == "production-rag"
    assert adapter.version == "unknown"


async def test_name_and_version_can_be_set():
    adapter = _adapter(FakeHttpClient(default=(200, {})), name="my-rag", version="abc123")
    assert adapter.name == "my-rag"
    assert adapter.version == "abc123"


# ---- request shape (matches the real AskRequest: question, top_k, rerank,
# optional document_id/conversation_id) ----

async def test_sends_question_top_k_and_rerank_to_ask_endpoint_with_api_key_header():
    http = FakeHttpClient(default=(200, {"answer": "30 days."}))
    adapter = _adapter(http)
    await adapter.run(TC)
    assert len(http.calls) == 1
    call = http.calls[0]
    assert call["url"] == "http://localhost:8000/ask"
    assert call["json"] == {"question": "What is the refund policy?", "top_k": 5, "rerank": True}
    assert call["headers"]["X-API-Key"] == "test-api-key"


async def test_top_k_and_rerank_constructor_defaults_are_sent():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = _adapter(http, top_k=3, rerank=False)
    await adapter.run(TC)
    assert http.calls[0]["json"]["top_k"] == 3
    assert http.calls[0]["json"]["rerank"] is False


async def test_per_test_case_metadata_overrides_top_k_and_rerank():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = _adapter(http)
    tc = TestCase(test_id="t2", input="x", metadata={"top_k": 2, "rerank": False})
    await adapter.run(tc)
    assert http.calls[0]["json"]["top_k"] == 2
    assert http.calls[0]["json"]["rerank"] is False


async def test_custom_ask_path_and_header_name():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = _adapter(http, ask_path="/v2/ask", api_key_header="Authorization")
    await adapter.run(TC)
    assert http.calls[0]["url"] == "http://localhost:8000/v2/ask"
    assert "Authorization" in http.calls[0]["headers"]


async def test_base_url_trailing_slash_normalized():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = ProductionRagAdapter(http, "http://localhost:8000/", "key")
    await adapter.run(TC)
    assert http.calls[0]["url"] == "http://localhost:8000/ask"


async def test_conversation_id_forwarded_when_present_in_metadata():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = _adapter(http)
    tc = TestCase(test_id="t2", input="follow-up question", metadata={"conversation_id": "11111111-1111-1111-1111-111111111111"})
    await adapter.run(tc)
    assert http.calls[0]["json"]["conversation_id"] == "11111111-1111-1111-1111-111111111111"


async def test_document_id_forwarded_when_present_in_metadata():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = _adapter(http)
    tc = TestCase(test_id="t2", input="x", metadata={"document_id": "22222222-2222-2222-2222-222222222222"})
    await adapter.run(tc)
    assert http.calls[0]["json"]["document_id"] == "22222222-2222-2222-2222-222222222222"


async def test_no_optional_keys_when_absent():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = _adapter(http)
    await adapter.run(TC)
    assert "conversation_id" not in http.calls[0]["json"]
    assert "document_id" not in http.calls[0]["json"]


# ---- response mapping: real AskResponse shape ----

async def test_maps_answer_and_citations_preserving_real_citation_shape():
    data = {
        "answer": "Refunds are available within 30 days.",
        "citations": [_citation()],
        "cost": {"total_cost_usd": 0.000463},
        "models": {"generation": "gpt-4o-mini", "embedding": "text-embedding-3-small"},
    }
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.output == "Refunds are available within 30 days."
    # The real Citation shape is preserved verbatim, not remapped.
    assert out.citations == [_citation()]
    assert out.context == ["Refunds within 30 days of purchase."]
    assert out.estimated_cost == 0.000463
    assert out.model == "gpt-4o-mini"  # the generation model specifically
    assert out.metadata["production_rag_models"] == data["models"]
    assert out.metadata["refused"] is False


async def test_maps_insufficient_evidence_refusal():
    data = {"answer": "", "insufficient_evidence": True, "refusal_reason": "no_results"}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.metadata["refused"] is True
    assert out.metadata["refusal_reason"] == "no_results"


async def test_maps_degraded_and_warnings_into_metadata():
    data = {"answer": "x", "degraded": True, "warnings": ["hybrid search fell back to keyword-only"]}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.metadata["degraded"] is True
    assert out.metadata["warnings"] == ["hybrid search fell back to keyword-only"]


async def test_no_degraded_or_warnings_keys_when_absent_or_empty():
    data = {"answer": "x", "degraded": False, "warnings": []}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert "degraded" not in out.metadata
    assert "warnings" not in out.metadata


async def test_search_query_surfaced_in_metadata_only_when_rewritten():
    data = {"answer": "x", "question": "orig", "search_query": "rewritten standalone query"}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.metadata["search_query"] == "rewritten standalone query"


async def test_search_query_not_surfaced_when_identical_to_question():
    data = {"answer": "x", "question": "same", "search_query": "same"}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert "search_query" not in out.metadata


async def test_maps_token_usage_summed_across_cost_breakdown_fields():
    data = {
        "answer": "x",
        "cost": {
            "embedding_tokens": 20,
            "rerank_prompt_tokens": 100, "rerank_completion_tokens": 10,
            "generation_prompt_tokens": 300, "generation_completion_tokens": 40,
            "contextualizer_prompt_tokens": 0, "contextualizer_completion_tokens": 0,
            "memory_extraction_prompt_tokens": 0, "memory_extraction_completion_tokens": 0,
            "memory_embedding_tokens": 0,
            "total_cost_usd": 0.001,
        },
    }
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.token_usage.input_tokens == 20 + 100 + 300  # embedding + rerank_prompt + generation_prompt
    assert out.token_usage.output_tokens == 10 + 40  # rerank_completion + generation_completion


async def test_no_token_usage_when_cost_breakdown_entirely_absent():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.token_usage is None
    assert out.estimated_cost is None


async def test_latency_ms_is_adapters_own_measurement_not_internal_timings():
    # production-rag's own timings_ms is preserved in metadata, but
    # SystemOutput.latency_ms is this adapter's OWN wall-clock measurement
    # -- see the adapter's module docstring for why.
    data = {"answer": "x", "timings_ms": {"retrieval": 50.0, "generation": 900.0}}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.latency_ms is not None and out.latency_ms >= 0
    assert out.metadata["production_rag_timings_ms"] == {"retrieval": 50.0, "generation": 900.0}


async def test_citations_without_snippet_dont_populate_context():
    data = {"answer": "x", "citations": [_citation(snippet=None)]}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.citations == [_citation(snippet=None)]
    assert out.context == []


# ---- malformed / partial citations: must never crash the adapter ----

async def test_malformed_citation_entries_are_dropped_not_crashed_on():
    data = {"answer": "x", "citations": [_citation(), "not-a-dict", 42, None]}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.citations == [_citation()]


async def test_citation_missing_expected_keys_is_kept_as_is():
    # A citation missing page_number entirely (a malformed/partial
    # production-rag response) must not crash the adapter -- it's kept
    # verbatim; an evaluator matching on page_number simply won't match it.
    partial = {"source_id": "S1", "chunk_id": "c-001"}
    data = {"answer": "x", "citations": [partial]}
    http = FakeHttpClient(default=(200, data))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.citations == [partial]
    assert out.context == []  # no snippet key -- correctly excluded, not crashed on


async def test_citations_entirely_missing_from_response():
    http = FakeHttpClient(default=(200, {"answer": "x"}))
    adapter = _adapter(http)
    out = await adapter.run(TC)
    assert out.citations == []
    assert out.context == []


# ---- error handling ----

async def test_http_error_status_raises_adapter_error():
    http = FakeHttpClient(default=(500, {"detail": "internal error"}))
    adapter = _adapter(http)
    with pytest.raises(ProductionRagAdapterError, match="HTTP 500"):
        await adapter.run(TC)


async def test_auth_error_status_raises_adapter_error():
    http = FakeHttpClient(default=(401, {"detail": "Missing API key. Provide 'Authorization: Bearer <key>' or 'X-API-Key: <key>'."}))
    adapter = _adapter(http)
    with pytest.raises(ProductionRagAdapterError, match="HTTP 401"):
        await adapter.run(TC)


async def test_embedding_provider_failure_status_raises_adapter_error():
    # production-rag's own router raises 502 for a live EmbeddingError and
    # 503 for a missing-API-key EmbeddingConfigError (app/routers/ask.py).
    http = FakeHttpClient(default=(502, {"detail": "Embedding provider failed; try again later."}))
    adapter = _adapter(http)
    with pytest.raises(ProductionRagAdapterError, match="HTTP 502"):
        await adapter.run(TC)


async def test_transport_failure_propagates_as_http_error():
    # A genuine crash (connection refused, DNS failure, a timeout) -- the
    # runner's own failure isolation (Day 5) is responsible for turning
    # this into a DEPENDENCY_FAILURE result, not this adapter.
    http = FakeHttpClient(raises=HttpError("connection refused"))
    adapter = _adapter(http)
    with pytest.raises(HttpError, match="connection refused"):
        await adapter.run(TC)


async def test_malformed_json_body_from_server_propagates_as_http_error():
    # HttpxHttpClient.post itself raises HttpError for a non-JSON body
    # (providers/http.py) -- simulated here via a FakeHttpClient configured
    # to raise the same way.
    http = FakeHttpClient(raises=HttpError("response was not valid JSON"))
    adapter = _adapter(http)
    with pytest.raises(HttpError, match="not valid JSON"):
        await adapter.run(TC)
