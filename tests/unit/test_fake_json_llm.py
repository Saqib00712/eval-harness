import pytest

from eval_harness.providers.llm import FakeJsonLLM


async def test_single_response_repeats_for_every_call():
    llm = FakeJsonLLM(responses={"score": 8, "reasoning": "good"})
    d1, _ = await llm.complete_json(model="m", system="s", user="u1", schema_name="x", schema={}, max_tokens=10)
    d2, _ = await llm.complete_json(model="m", system="s", user="u2", schema_name="x", schema={}, max_tokens=10)
    assert d1 == d2 == {"score": 8, "reasoning": "good"}


async def test_list_of_responses_consumed_in_order():
    llm = FakeJsonLLM(responses=[{"score": 1, "reasoning": "a"}, {"score": 9, "reasoning": "b"}])
    d1, _ = await llm.complete_json(model="m", system="s", user="u", schema_name="x", schema={}, max_tokens=10)
    d2, _ = await llm.complete_json(model="m", system="s", user="u", schema_name="x", schema={}, max_tokens=10)
    assert d1["score"] == 1 and d2["score"] == 9


async def test_list_exhausted_raises_index_error():
    llm = FakeJsonLLM(responses=[{"score": 1, "reasoning": "a"}])
    await llm.complete_json(model="m", system="s", user="u", schema_name="x", schema={}, max_tokens=10)
    with pytest.raises(IndexError):
        await llm.complete_json(model="m", system="s", user="u", schema_name="x", schema={}, max_tokens=10)


async def test_default_used_when_no_responses_scripted():
    llm = FakeJsonLLM(default={"score": 5, "reasoning": "default"})
    d, _ = await llm.complete_json(model="m", system="s", user="u", schema_name="x", schema={}, max_tokens=10)
    assert d["score"] == 5


async def test_no_responses_and_no_default_raises():
    llm = FakeJsonLLM()
    with pytest.raises(IndexError):
        await llm.complete_json(model="m", system="s", user="u", schema_name="x", schema={}, max_tokens=10)


async def test_calls_are_recorded():
    llm = FakeJsonLLM(default={"score": 5, "reasoning": "x"})
    await llm.complete_json(model="gpt-test", system="sys", user="usr", schema_name="sch", schema={}, max_tokens=10)
    assert llm.calls == [{"model": "gpt-test", "system": "sys", "user": "usr", "schema_name": "sch"}]


async def test_default_usage_is_nonzero():
    llm = FakeJsonLLM(default={"score": 5, "reasoning": "x"})
    _, usage = await llm.complete_json(model="m", system="s", user="u", schema_name="x", schema={}, max_tokens=10)
    assert usage.total_tokens > 0
