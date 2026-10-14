"""
MockSystem: a SystemUnderTest with no real AI behind it at all -- canned
SystemOutputs keyed by test_id. This is a first-class citizen of the
package (not test-only scaffolding): Phase 3 explicitly lists "Mock
application" as one of the system kinds the harness must support, and
later days (the runner, Phase 32's demonstration, this project's own test
suite) all need a system that is deterministic, free, and instant.
"""

from eval_harness.models import TestCase
from eval_harness.system_output import SystemOutput


class MockSystem:
    def __init__(
        self,
        responses: dict[str, SystemOutput],
        *,
        name: str = "mock-system",
        version: str = "0.0.0",
        default: SystemOutput | None = None,
    ):
        """`responses` maps test_id -> the SystemOutput to return for that
        exact test case. `default`, if given, is returned for any test_id
        not in `responses` instead of raising -- useful when you only care
        about canning a few specific cases and want everything else to
        behave like "the system had nothing to say"."""
        self._responses = dict(responses)
        self._name = name
        self._version = version
        self._default = default

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    async def run(self, test_case: TestCase) -> SystemOutput:
        if test_case.test_id in self._responses:
            return self._responses[test_case.test_id]
        if self._default is not None:
            return self._default
        raise KeyError(
            f"MockSystem {self._name!r} has no canned response for test_id={test_case.test_id!r} "
            f"and no default was configured."
        )
