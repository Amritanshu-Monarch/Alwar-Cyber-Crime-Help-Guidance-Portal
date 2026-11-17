"""Unit tests for Phase 5: Golden Dataset and Evaluation Pipeline."""

import json
from pathlib import Path

import pytest

from evaluation.eval_runner import check_hallucinated_numbers, evaluate_query


def test_golden_dataset_structure_and_coverage():
    """Verify that golden_dataset.json contains at least 100 cases across all required categories."""
    dataset_path = Path("backend/tests/eval/golden_dataset.json")
    assert dataset_path.exists(), "golden_dataset.json must exist in backend/tests/eval/"

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) >= 100, f"Expected at least 100 golden cases, got {len(cases)}"

    categories = {c["category"] for c in cases}
    required_categories = {
        "upi_fraud",
        "sextortion",
        "deepfake",
        "account_hack",
        "lost_phone",
        "digital_arrest",
        "loan_app",
        "thana_finder",
        "out_of_scope",
    }
    assert required_categories.issubset(categories), f"Missing categories: {required_categories - categories}"

    languages = {c["lang"] for c in cases}
    assert {"en", "hi", "hinglish"}.issubset(languages), f"Missing languages: {{'en', 'hi', 'hinglish'}} - {languages}"

    for case in cases:
        assert "id" in case
        assert "query" in case
        assert "expected_facts" in case
        assert "forbidden_facts" in case
        assert "expected_scope" in case


def test_check_hallucinated_numbers_detector():
    """Verify detection of unverified phone numbers in official police contact contexts."""
    # Official contact context with unverified number -> True
    assert check_hallucinated_numbers("Call SHO at 9988776655 for police help", "Help me") is True

    # User provided number in query -> False (preserved)
    assert check_hallucinated_numbers("Suspect called from 9876543210", "Fraud from 9876543210") is False

    # UTR or Txn ID -> False (preserved)
    assert check_hallucinated_numbers("Your UTR 9876543210 is registered", "My money lost") is False


@pytest.mark.anyio
async def test_evaluate_query_mock_execution():
    """Verify that evaluate_query executes cleanly without throwing exceptions."""
    sample_case = {
        "id": "tc_test_001",
        "category": "upi_fraud",
        "lang": "en",
        "query": "Rs 10,000 lost in UPI scam via QR code.",
        "expected_facts": ["1930", "bank"],
        "forbidden_facts": ["fee"],
        "expected_scope": "in_scope",
    }
    res = await evaluate_query(sample_case, service=None, mock=True)
    assert res.case_id == "tc_test_001"
    assert res.is_hallucinated_number is False
    assert res.is_false_reject is False
    assert res.latency_s > 0
    assert len(res.reply) > 0
