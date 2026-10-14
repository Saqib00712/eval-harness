"""
Phase 6: deterministic evaluators -- cheap, I/O-free, no LLM call, no
embedding call, 100% reproducible (same input always gives the same
score). Phase 21 says these should run BEFORE any LLM judge in a pipeline,
precisely because they're free and instant; the runner (Day 5) will take
that ordering as its default.

Configuration pattern shared by every evaluator here: a fixed value can be
set once at construction (applies to every test case this evaluator
instance scores -- convenient when the same rule applies dataset-wide,
e.g. "every answer must be under 500 characters"), OR given per-test-case
via `test_case.metadata` under a documented key (overrides the constructor
default for that one case -- needed when different questions need
different checks, e.g. each question has its own expected regex). The
metadata value wins when both are set. If NEITHER is set, `evaluate()`
raises `ValueError` rather than silently returning a 0.0 score -- a
misconfigured evaluator should fail loudly during a run, not produce a
number indistinguishable from a real failure.
"""

import re

import jsonschema

from eval_harness.evaluator import EvaluatorResult
from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput


class ExactMatchEvaluator:
    """Passes iff `output.output` equals `test_case.expected_output`, after
    normalization (surrounding whitespace stripped; case-folded unless
    `case_sensitive=True`)."""

    def __init__(self, *, case_sensitive: bool = False):
        self._case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "exact_match"

    def _normalize(self, s: str) -> str:
        s = s.strip()
        return s if self._case_sensitive else s.lower()

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        expected = test_case.expected_output
        if expected is None:
            raise ValueError(
                f"{self.name}: test case {test_case.test_id!r} has no expected_output to compare against."
            )
        actual = output.output or ""
        passed = self._normalize(actual) == self._normalize(expected)
        return EvaluatorResult(
            evaluator=self.name, score=1.0 if passed else 0.0, threshold=1.0, passed=passed,
            reason="Exact match." if passed else f"Expected {expected!r}, got {actual!r}.",
        )


class ContainsEvaluator:
    """Checks that required terms appear in `output.output` (case-insensitive
    substring match). Terms: constructor `terms`, overridden per-case by
    `test_case.metadata['required_terms']`. `require_all=True` (default)
    needs every term present to pass; `require_all=False` needs at least
    one. `score` is always the fraction of terms found, regardless of mode."""

    def __init__(self, *, terms: list[str] | None = None, require_all: bool = True):
        self._terms = terms
        self._require_all = require_all

    @property
    def name(self) -> str:
        return "contains"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        terms = test_case.metadata.get("required_terms") or self._terms
        if not terms:
            raise ValueError(f"{self.name}: no required terms configured for test case {test_case.test_id!r}.")
        text = (output.output or "").lower()
        hits = [t for t in terms if t.lower() in text]
        missing = [t for t in terms if t not in hits]
        passed = (len(hits) == len(terms)) if self._require_all else (len(hits) > 0)
        score = round(len(hits) / len(terms), 4)
        reason = "All required terms present." if passed else f"Missing: {missing}"
        return EvaluatorResult(
            evaluator=self.name, score=score, threshold=1.0 if self._require_all else None, passed=passed,
            reason=reason, metadata={"hits": hits, "missing": missing},
        )


class RegexEvaluator:
    """Passes iff `output.output` matches (via `re.search`) a pattern.
    Pattern: constructor `pattern`, overridden per-case by
    `test_case.metadata['pattern']`. Useful for structured-ish free text
    (e.g. an expected date format, a required prefix).

    SECURITY (Day 17): both `pattern` and the matched text can come from a
    loaded dataset -- untrusted input, per loader.py's own ceilings this
    same day added. A pathological pattern (nested quantifiers like
    `(a+)+$`) matched against crafted input can trigger catastrophic
    backtracking (ReDoS) and hang the Python process for a very long time;
    `re`'s backtracking engine has no built-in timeout. `max_pattern_length`
    / `max_text_length` block the CHEAPEST version of this attack (the
    length a pathological pattern or input needs to become slow grows
    quickly, so a tight length cap rules out most real payloads) but this
    is a deliberately partial mitigation, not a guarantee: a short pattern
    can still be pathological. There is no fully portable fix available
    here without a third-party dependency (the `regex` package supports a
    real per-call timeout; stdlib `re` does not, and a signal-based
    timeout does not work on Windows, which this harness explicitly
    supports). Treat regex patterns from a dataset you did not author
    yourself as you would any other untrusted input: review them, or keep
    these caps low, before running them in CI."""

    def __init__(
        self, *, pattern: str | None = None, flags: int = 0,
        max_pattern_length: int = 200, max_text_length: int = 10_000,
    ):
        self._pattern = pattern
        self._flags = flags
        self._max_pattern_length = max_pattern_length
        self._max_text_length = max_text_length

    @property
    def name(self) -> str:
        return "regex"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        pattern = test_case.metadata.get("pattern") or self._pattern
        if not pattern:
            raise ValueError(f"{self.name}: no pattern configured for test case {test_case.test_id!r}.")
        if len(pattern) > self._max_pattern_length:
            raise ValueError(
                f"{self.name}: pattern is {len(pattern)} chars, over the {self._max_pattern_length}-char "
                f"limit (a cheap first line of defense against catastrophic-backtracking regex payloads)."
            )
        text = output.output or ""
        if len(text) > self._max_text_length:
            raise ValueError(
                f"{self.name}: matched text is {len(text)} chars, over the {self._max_text_length}-char limit."
            )
        match = re.search(pattern, text, self._flags)
        passed = match is not None
        return EvaluatorResult(
            evaluator=self.name, score=1.0 if passed else 0.0, threshold=1.0, passed=passed,
            reason=f"Pattern {pattern!r} {'matched' if passed else 'did not match'}.",
            metadata={"matched_text": match.group(0) if match else None},
        )


class JSONSchemaEvaluator:
    """Validates `output.structured_data` against a JSON Schema (via the
    `jsonschema` package -- real Draft 2020-12 validation, not a hand-rolled
    approximation). Schema: constructor `schema`, overridden per-case by
    `test_case.metadata['json_schema']`."""

    def __init__(self, *, schema: dict | None = None):
        self._schema = schema

    @property
    def name(self) -> str:
        return "json_schema"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        schema = test_case.metadata.get("json_schema") or self._schema
        if not schema:
            raise ValueError(f"{self.name}: no JSON schema configured for test case {test_case.test_id!r}.")
        data = output.structured_data
        if data is None:
            return EvaluatorResult(
                evaluator=self.name, score=0.0, threshold=1.0, passed=False,
                reason="output.structured_data is None -- nothing to validate.",
            )
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            return EvaluatorResult(
                evaluator=self.name, score=0.0, threshold=1.0, passed=False,
                reason=f"Schema validation failed: {exc.message}",
            )
        return EvaluatorResult(evaluator=self.name, score=1.0, threshold=1.0, passed=True, reason="Valid against schema.")


class LengthEvaluator:
    """Passes iff the length of `output.output` (characters, or words if
    `unit='words'`) falls within `[min_length, max_length]`. Either bound
    may be left `None` (unbounded on that side). Bounds: constructor
    defaults, overridden per-case by `test_case.metadata['min_length']` /
    `['max_length']`."""

    def __init__(self, *, min_length: int | None = None, max_length: int | None = None, unit: str = "chars"):
        if unit not in ("chars", "words"):
            raise ValueError("unit must be 'chars' or 'words'")
        self._min = min_length
        self._max = max_length
        self._unit = unit

    @property
    def name(self) -> str:
        return "length"

    def _measure(self, text: str) -> int:
        return len(text.split()) if self._unit == "words" else len(text)

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        min_length = test_case.metadata.get("min_length", self._min)
        max_length = test_case.metadata.get("max_length", self._max)
        if min_length is None and max_length is None:
            raise ValueError(f"{self.name}: no min_length/max_length configured for test case {test_case.test_id!r}.")
        length = self._measure(output.output or "")
        too_short = min_length is not None and length < min_length
        too_long = max_length is not None and length > max_length
        passed = not (too_short or too_long)
        if too_short:
            reason = f"Length {length} {self._unit} is below minimum {min_length}."
        elif too_long:
            reason = f"Length {length} {self._unit} exceeds maximum {max_length}."
        else:
            reason = f"Length {length} {self._unit} is within bounds."
        return EvaluatorResult(
            evaluator=self.name, score=1.0 if passed else 0.0, threshold=1.0, passed=passed,
            reason=reason, metadata={"length": length, "unit": self._unit},
        )


class RequiredFieldsEvaluator:
    """Passes iff every required key is present AND not `None` in
    `output.structured_data`. Fields: constructor `fields`, overridden
    per-case by `test_case.metadata['required_fields']`."""

    def __init__(self, *, fields: list[str] | None = None):
        self._fields = fields

    @property
    def name(self) -> str:
        return "required_fields"

    async def evaluate(self, test_case: TestCase, output: SystemOutput) -> EvaluatorResult:
        fields = test_case.metadata.get("required_fields") or self._fields
        if not fields:
            raise ValueError(f"{self.name}: no required_fields configured for test case {test_case.test_id!r}.")
        data = output.structured_data or {}
        missing = [f for f in fields if f not in data or data[f] is None]
        passed = not missing
        score = round((len(fields) - len(missing)) / len(fields), 4)
        reason = "All required fields present." if passed else f"Missing fields: {missing}"
        return EvaluatorResult(
            evaluator=self.name, score=score, threshold=1.0, passed=passed, reason=reason,
            metadata={"missing": missing},
        )
