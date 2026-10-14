"""
Pure retrieval-quality metrics. No I/O, no Pydantic, no embeddings -- unit
testable with plain lists, exactly why Project 1 kept its own version of
these (app/eval/retrieval_metrics.py) separate from its runner.

This is a deliberate GENERALIZATION of that pattern, not a copy-paste: Project
1's functions are keyed on `int` page numbers (`hit_pages: list[int]`,
`expected_pages: list[int]`) because Project 1 only ever retrieves PDF pages.
This harness has to work for any system -- a RAG app retrieving pages, a
different app retrieving chunk IDs, URLs, or raw passage text -- so every
function here is keyed on a generic, hashable "retrieved item" (`str` in
practice: SystemOutput.context entries), never on a page number specifically.
The math is identical; only the identifier type changed.

Definitions (the interview-ready version -- same ones Project 1 uses):
  hit@k    - did AT LEAST ONE expected item appear in the top k retrieved
             items? Binary per-question; averaged across a dataset it's a
             "success rate". Cheap to compute, easy to explain.
  MRR      - Mean Reciprocal Rank: 1 / (rank of the FIRST expected item
             found), 0 if never found. Rewards finding the answer EARLY, not
             just eventually -- a hit at rank 1 scores 1.0, at rank 10 scores
             0.1. Better than hit@k at distinguishing "great" from "barely
             adequate" retrieval.
  recall@k - of ALL expected items for a question, what fraction appeared in
             the top k? Matters when an answer is spread across several
             retrieved items; hit@k alone would call that a full success at
             just one of them.

Matching is by EXACT equality of the retrieved item against the expected
item (exact string match, in practice) -- the same simplification Project 1
makes with exact page-number equality. A retrieved chunk that paraphrases or
overlaps an expected one without matching it exactly is scored as a miss
here; that's a real limitation (worth a smarter fuzzy/overlap matcher some
day), not an oversight -- it's documented so it isn't mistaken for a bug.
"""


def compute_ranks(retrieved: list[str], expected: list[str]) -> dict[str, int]:
    """For each expected item, the EARLIEST 1-indexed rank it appeared at in
    `retrieved`. Expected items never found are simply absent from the
    result (not an error -- that's what a miss looks like)."""
    ranks: dict[str, int] = {}
    expected_set = set(expected)
    for rank, item in enumerate(retrieved, start=1):
        if item in expected_set and item not in ranks:
            ranks[item] = rank
    return ranks


def hit_at_k(item_ranks: dict[str, int], k: int) -> bool:
    return any(r <= k for r in item_ranks.values())


def reciprocal_rank(item_ranks: dict[str, int]) -> float:
    if not item_ranks:
        return 0.0
    return 1.0 / min(item_ranks.values())


def recall_at_k(item_ranks: dict[str, int], expected: list[str], k: int) -> float:
    if not expected:
        # A question with nothing expected to retrieve (e.g. an
        # unanswerable question, correctly scored on refusal elsewhere, not
        # here) has nothing to recall. 0.0, not 1.0 -- returning 1.0 would
        # misleadingly count as "perfect retrieval" for a query that was
        # never supposed to retrieve anything. Same convention Project 1
        # uses for its own recall_at_k.
        return 0.0
    found = {item for item, r in item_ranks.items() if r <= k}
    return len(found) / len(set(expected))
