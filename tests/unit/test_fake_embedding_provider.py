from eval_harness.providers.embedding import EmbeddingProvider, FakeEmbeddingProvider
from eval_harness.similarity import cosine_similarity


def test_satisfies_the_protocol():
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)


async def test_identical_text_is_perfectly_similar():
    provider = FakeEmbeddingProvider()
    vectors = await provider.embed(["Thirty days.", "Thirty days."])
    assert cosine_similarity(vectors[0], vectors[1]) == 1.0


async def test_is_deterministic_across_separate_calls():
    provider = FakeEmbeddingProvider()
    a = await provider.embed(["Thirty days."])
    b = await provider.embed(["Thirty days."])
    assert a[0] == b[0]


async def test_near_duplicate_text_scores_highly_similar():
    provider = FakeEmbeddingProvider()
    vectors = await provider.embed(["The refund window is 30 days.", "The refund window is 30 day."])
    assert cosine_similarity(vectors[0], vectors[1]) > 0.9


async def test_unrelated_text_scores_lower_than_near_duplicates():
    provider = FakeEmbeddingProvider()
    vectors = await provider.embed(["The refund window is 30 days.", "Mount Everest is the tallest mountain."])
    score = cosine_similarity(vectors[0], vectors[1])
    assert score < 0.5


def test_empty_string_does_not_crash():
    provider = FakeEmbeddingProvider()
    vec = provider._embed_one("")
    assert len(vec) == provider._dims
