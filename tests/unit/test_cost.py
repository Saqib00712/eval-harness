from eval_harness.cost import estimate_cost_usd


def test_estimate_cost_usd_basic():
    assert estimate_cost_usd(1_000_000, 5.0) == 5.0


def test_estimate_cost_usd_fractional():
    assert estimate_cost_usd(500_000, 5.0) == 2.5


def test_estimate_cost_usd_zero_tokens():
    assert estimate_cost_usd(0, 5.0) == 0.0
