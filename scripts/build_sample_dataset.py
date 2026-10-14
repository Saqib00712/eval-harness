"""
Phase 31: build the 20-example synthetic sample dataset and save it to
datasets/sample_eval_v1.json.

Why a script instead of a hand-written JSON file: generating it through
`Dataset(...)` + `save_dataset(...)` guarantees the file is always valid
against the current schema (and always carries a correctly-computed
content_hash) -- the same reproducibility guarantee the harness asks every
dataset to have. Re-run this any time the sample set needs to change:

    python scripts/build_sample_dataset.py

All content below is synthetic (an invented company, "TechNova Corp", and
an invented policy handbook) -- no real personal data, per Phase 31.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from eval_harness.dataset import Dataset  # noqa: E402
from eval_harness.dataset_io import save_dataset  # noqa: E402
from eval_harness.models import TestCase  # noqa: E402

# Each test case is tagged (metadata.category + tags) with the KIND of
# challenge it is designed to pose to a system under test -- not every
# category maps to a "wrong answer" in the golden data itself (the golden
# data always describes the CORRECT behavior); the tag says what a WEAK
# system is likely to get wrong on that question. This lets later metrics
# group pass/fail rates by category (Phase 14: "citation_error: 94%
# pass rate") instead of only reporting one flat number.
CASES = [
    TestCase(
        test_id="qa-001", input="What is the refund policy?",
        expected_output="Customers can request a full refund within 30 days of purchase.",
        reference_context=["Section 4: Refunds -- customers may request a full refund within 30 days of purchase."],
        expected_citations=[{"page": 4}],
        metadata={"category": "easy_correct"}, tags=["refund", "policy"],
    ),
    TestCase(
        test_id="qa-002", input="What is the CEO's personal phone number?",
        expected_output=None,  # should be refused -- not in the document, and not the kind of thing a RAG system should ever answer
        reference_context=[], expected_citations=[],
        metadata={"category": "unanswerable_should_refuse"}, tags=["refusal", "out_of_scope"],
    ),
    TestCase(
        test_id="qa-003", input="On what page is the data retention policy described?",
        expected_output="Data is retained for 90 days after account closure, described on page 7.",
        reference_context=["Section 7: Data Retention -- account data is retained for 90 days after closure."],
        expected_citations=[{"page": 7}],
        metadata={"category": "citation_precision"}, tags=["data_retention", "policy"],
    ),
    TestCase(
        test_id="qa-004", input="What are all the support channels available?",
        expected_output="Email, live chat, and phone support (Enterprise plan only).",
        reference_context=[
            "Section 2: Support -- email and live chat are available on all plans.",
            "Section 3: Enterprise Support -- phone support is available exclusively on the Enterprise plan.",
        ],
        expected_citations=[{"page": 2}, {"page": 3}],
        metadata={"category": "multi_page_aggregation"}, tags=["support"],
    ),
    TestCase(
        test_id="qa-005", input="What is the maximum file upload size?",
        expected_output="25 GB per file.",
        reference_context=["Section 5: Storage Limits -- the maximum upload size is 25 GB per file."],
        expected_citations=[{"page": 5}],
        metadata={"category": "numeric_precision"}, tags=["storage", "limits"],
    ),
    TestCase(
        test_id="qa-006", input="Is the free tier eligible for phone support?",
        expected_output="No, phone support is only available on the Enterprise plan.",
        reference_context=["Section 3: Enterprise Support -- phone support is available exclusively on the Enterprise plan."],
        expected_citations=[{"page": 3}],
        metadata={"category": "negation_tricky"}, tags=["support", "pricing"],
    ),
    TestCase(
        test_id="qa-007", input="What is the weather in Lahore today?",
        expected_output=None,  # should be refused -- entirely unrelated to the corpus
        reference_context=[], expected_citations=[],
        metadata={"category": "irrelevant_to_corpus"}, tags=["refusal", "irrelevant"],
    ),
    TestCase(
        test_id="qa-008", input="What does 'SLA' stand for in this document?",
        expected_output="Service Level Agreement.",
        reference_context=["Section 6: SLA (Service Level Agreement) -- defines our uptime commitment."],
        expected_citations=[{"page": 6}],
        metadata={"category": "definition_lookup"}, tags=["sla", "glossary"],
    ),
    TestCase(
        test_id="qa-009", input="How much does it cost?",
        expected_output="It depends on the plan: Starter, Pro, and Enterprise each have different monthly prices (see Section 8).",
        reference_context=["Section 8: Pricing -- three tiers are offered: Starter, Pro, and Enterprise."],
        expected_citations=[{"page": 8}],
        metadata={"category": "ambiguous_question"}, tags=["pricing"],
    ),
    TestCase(
        test_id="qa-010", input="How many pricing tiers are offered?",
        expected_output="Three: Starter, Pro, and Enterprise.",
        reference_context=["Section 8: Pricing -- three tiers are offered: Starter, Pro, and Enterprise."],
        expected_citations=[{"page": 8}],
        metadata={"category": "short_factoid"}, tags=["pricing"],
    ),
    TestCase(
        test_id="qa-011", input="What happens if I cancel on day 30 of the refund window?",
        expected_output="Cancellations on day 30 are still eligible for a full refund; day 31 onward is not.",
        reference_context=["Section 4: Refunds -- the 30-day window is inclusive of day 30; requests from day 31 onward are not eligible."],
        expected_citations=[{"page": 4}],
        metadata={"category": "edge_case_boundary"}, tags=["refund", "policy"],
    ),
    TestCase(
        test_id="qa-012", input="How long does it take to receive a refund after approval?",
        expected_output="5-7 business days.",
        reference_context=["Section 4: Refunds -- approved refunds are processed within 5-7 business days."],
        expected_citations=[{"page": 4}],
        metadata={"category": "temporal_sensitive"}, tags=["refund"],
    ),
    TestCase(
        test_id="qa-013", input="If a customer on the Starter plan requests a refund after 45 days, are they eligible?",
        expected_output="No -- the refund window is 30 days regardless of plan, so a 45-day-old request is not eligible.",
        reference_context=["Section 4: Refunds -- customers may request a full refund within 30 days of purchase, on any plan."],
        expected_citations=[{"page": 4}],
        metadata={"category": "multi_hop_reasoning"}, tags=["refund", "pricing"],
    ),
    TestCase(
        test_id="qa-014", input="List the three pricing tiers and their monthly prices.",
        expected_output="Starter: $9/mo, Pro: $29/mo, Enterprise: custom pricing.",
        reference_context=["Section 8: Pricing -- Starter is $9/mo, Pro is $29/mo, and Enterprise uses custom pricing."],
        expected_citations=[{"page": 8}],
        metadata={"category": "structured_output_expected"}, tags=["pricing"],
    ),
    TestCase(
        test_id="qa-015", input="Can I get my money back?",
        expected_output="Customers can request a full refund within 30 days of purchase.",
        reference_context=["Section 4: Refunds -- customers may request a full refund within 30 days of purchase."],
        expected_citations=[{"page": 4}],
        metadata={"category": "synonym_paraphrase"}, tags=["refund"],
    ),
    TestCase(
        test_id="qa-016", input="What is the support email address?",
        expected_output="support@technova-example.com",
        reference_context=["Section 2: Support -- reach us by email at support@technova-example.com."],
        expected_citations=[{"page": 2}],
        metadata={"category": "format_sensitive"}, tags=["support"],
    ),
    TestCase(
        test_id="qa-017", input="Does TechNova offer on-premise deployment?",
        expected_output=None,  # should be refused -- the document never mentions on-premise deployment
        reference_context=[], expected_citations=[],
        metadata={"category": "out_of_scope_adjacent"}, tags=["refusal", "deployment"],
    ),
    TestCase(
        test_id="qa-018", input="Summarize the security practices described in the document.",
        expected_output="Data is encrypted at rest and in transit, and access is restricted via role-based permissions.",
        reference_context=[
            "Section 10: Security -- all data is encrypted at rest and in transit.",
            "Section 11: Access Control -- access is restricted using role-based permissions.",
        ],
        expected_citations=[{"page": 10}, {"page": 11}],
        metadata={"category": "long_context"}, tags=["security"],
    ),
    TestCase(
        test_id="qa-019", input="What is the uptime guarantee percentage?",
        expected_output="99.9%",
        reference_context=["Section 6: SLA (Service Level Agreement) -- we guarantee 99.9% uptime."],
        expected_citations=[{"page": 6}],
        metadata={"category": "numeric_precision"}, tags=["sla"],
    ),
    TestCase(
        test_id="qa-020", input="What's the policy on account deactivation due to inactivity?",
        expected_output="Accounts inactive for 12 months are automatically deactivated after a 30-day warning email.",
        reference_context=["Section 9: Account Lifecycle -- accounts inactive for 12 months are deactivated, following a 30-day warning email."],
        expected_citations=[{"page": 9}],
        metadata={"category": "retrieval_failure"}, tags=["account_lifecycle"],
    ),
]


def main() -> None:
    dataset = Dataset(
        dataset_id="sample_eval",
        name="TechNova Corp policy handbook -- sample synthetic eval set",
        version="1.0.0",
        description=(
            "20 synthetic question/answer pairs against an invented company's policy handbook, "
            "covering easy factoid lookups, refusal cases (unanswerable + irrelevant), multi-page "
            "aggregation, citation precision, negation, ambiguity, and multi-hop reasoning. "
            "Built for Phase 31 / Phase 32's quality-gate demonstration."
        ),
        test_cases=CASES,
    )
    save_dataset(dataset, "datasets/sample_eval_v1.json")
    print(f"Wrote datasets/sample_eval_v1.json -- {dataset.test_count} test cases, content_hash={dataset.content_hash[:12]}...")


if __name__ == "__main__":
    main()
