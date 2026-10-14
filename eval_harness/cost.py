"""Token -> USD accounting. Pure function, trivially testable -- a direct
reuse of Project 1's own app/services/cost.py, which already kept this as
one line of pure math separate from any HTTP/SDK code for exactly that
reason. Needed from Day 8 onward: an LLM-judge evaluator calls a real,
billed API and must be able to report its own cost (see evaluator.py's
`EvaluatorResult.estimated_cost`)."""


def estimate_cost_usd(tokens: int, price_per_million_tokens: float) -> float:
    return round(tokens / 1_000_000 * price_per_million_tokens, 8)
