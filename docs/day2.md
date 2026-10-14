# Day 2 — Dataset versioning + sample dataset

## Phase 2: dataset versioning

`eval_harness/hashing.py` — `compute_content_hash(test_cases)`: a pure, order-independent SHA-256 fingerprint of a list of `TestCase`s (rows are sorted by `test_id` and JSON-canonicalized before hashing, so two lists with the same content in a different order hash identically).

`eval_harness/dataset.py` — `Dataset`: wraps a named, versioned collection of test cases (`dataset_id`, `name`, `version`, `description`, `created_at`, `test_cases`). Two fields are **computed**, never set by hand:

- `test_count` — `len(test_cases)`
- `content_hash` — `compute_content_hash(test_cases)`

**Why computed, not stored-and-maintained:** a stored hash can go stale (you edit `test_cases`, forget to update the hash field next to it). A computed one can't — it's always freshly derived from whatever `test_cases` currently is. This is what makes "the same dataset must be reproducible" true by construction, not by discipline. Proven by `test_hand_edited_test_cases_change_hash_on_reload`: even if a hand-edited file's stored `content_hash` is left stale, reloading it never trusts that stale value — it's recomputed and will differ.

`version` stays separate and human-assigned (`"1.0.0"`, a date, whatever you want) — it answers "what a person called this", while `content_hash` answers "is this exactly the same data". An `EvaluationRun` (Day 11) will record both.

`eval_harness/dataset_io.py` — `save_dataset`/`load_dataset`: round-trips a `Dataset` to/from one JSON manifest file (metadata + `test_cases` together — distinct from Day 1's `loader.load_test_cases`, which reads a bare array with no dataset-level identity).

## Phase 31: sample dataset

`scripts/build_sample_dataset.py` generates `datasets/sample_eval_v1.json` — 20 synthetic Q&A pairs against an invented company's ("TechNova Corp") policy handbook. Built as a script (not hand-written JSON) so it's always valid against the current schema and always carries a correct hash.

Each case is tagged with the *kind* of challenge it poses (`metadata.category`): `easy_correct`, `unanswerable_should_refuse` ×2, `citation_precision`, `multi_page_aggregation`, `numeric_precision` ×2, `negation_tricky`, `irrelevant_to_corpus`, `definition_lookup`, `ambiguous_question`, `short_factoid`, `edge_case_boundary`, `temporal_sensitive`, `multi_hop_reasoning`, `structured_output_expected`, `synonym_paraphrase`, `format_sensitive`, `out_of_scope_adjacent`, `long_context`, `retrieval_failure` — 19 distinct categories across 20 cases. Later days (Phase 14's per-category metrics) will group pass rates by this tag.

4 refusal-worthy questions (2 unanswerable, 1 irrelevant, 1 out-of-scope-adjacent) are included deliberately — without them, the dataset could never catch a system that hallucinates instead of saying "I don't know."

## Tests

28 new tests (`test_hashing.py`, `test_dataset.py`, `test_dataset_io.py`, `test_sample_dataset.py`), all offline. **Total: 47 passed.**

## Next

Day 3 — Phase 3 + 4: the `SystemUnderTest` abstraction and `SystemOutput` model, plus a `MockSystem` so the runner (coming Day 5) is testable without any real AI system behind it yet.
