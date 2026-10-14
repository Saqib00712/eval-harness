from eval_harness.hashing import compute_content_hash
from eval_harness.models import TestCase

A = TestCase(test_id="qa-001", input="What is the refund policy?", expected_output="30 days")
B = TestCase(test_id="qa-002", input="What is the capital of France?", expected_output="Paris")


def test_hash_is_deterministic():
    assert compute_content_hash([A, B]) == compute_content_hash([A, B])


def test_hash_is_order_independent():
    assert compute_content_hash([A, B]) == compute_content_hash([B, A])


def test_hash_changes_with_content():
    changed = TestCase(test_id="qa-001", input="What is the refund policy?", expected_output="45 days")
    assert compute_content_hash([A, B]) != compute_content_hash([changed, B])


def test_empty_list_has_a_stable_hash():
    assert compute_content_hash([]) == compute_content_hash([])


def test_hash_is_a_sha256_hex_digest():
    h = compute_content_hash([A, B])
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
