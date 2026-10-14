from eval_harness.retrieval_metrics import compute_ranks, hit_at_k, recall_at_k, reciprocal_rank


def test_compute_ranks_finds_earliest_occurrence():
    ranks = compute_ranks(["a", "b", "a", "c"], ["a", "c"])
    assert ranks == {"a": 1, "c": 4}


def test_compute_ranks_omits_items_never_found():
    ranks = compute_ranks(["a", "b"], ["z"])
    assert ranks == {}


def test_hit_at_k_true_when_within_k():
    assert hit_at_k({"a": 3}, k=5) is True


def test_hit_at_k_false_when_beyond_k():
    assert hit_at_k({"a": 7}, k=5) is False


def test_hit_at_k_false_when_no_ranks():
    assert hit_at_k({}, k=5) is False


def test_reciprocal_rank_uses_earliest_rank():
    assert reciprocal_rank({"a": 4, "b": 2}) == 0.5


def test_reciprocal_rank_zero_when_nothing_found():
    assert reciprocal_rank({}) == 0.0


def test_recall_at_k_fraction_within_window():
    ranks = compute_ranks(["a", "b", "c", "d"], ["a", "c", "z"])
    assert recall_at_k(ranks, ["a", "c", "z"], k=3) == 2 / 3


def test_recall_at_k_zero_when_nothing_expected():
    assert recall_at_k({}, [], k=5) == 0.0


def test_recall_at_k_respects_k_window():
    ranks = compute_ranks(["a", "b", "c"], ["c"])
    assert recall_at_k(ranks, ["c"], k=2) == 0.0
    assert recall_at_k(ranks, ["c"], k=3) == 1.0
