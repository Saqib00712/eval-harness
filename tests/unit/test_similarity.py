import pytest

from eval_harness.similarity import cosine_similarity


def test_identical_vectors_score_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_score_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_magnitude_does_not_affect_similarity():
    """Cosine similarity ignores vector length -- scaling a vector must not
    change the result."""
    a = [1.0, 2.0, 3.0]
    scaled = [2.0, 4.0, 6.0]
    assert cosine_similarity(a, scaled) == pytest.approx(1.0)


def test_zero_vector_scores_zero_not_an_error():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_mismatched_length_raises():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
