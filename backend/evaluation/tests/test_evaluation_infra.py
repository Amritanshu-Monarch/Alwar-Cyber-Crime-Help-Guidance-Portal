"""Tests for Sub-phase B3.4.0: Evaluation Framework Architecture & Installation.

Verifies:
1. Reusable evaluation dataset loading and schema validation
2. Sensitive PII / credentials redaction
3. Custom deterministic evaluation metrics
4. DeepEval adapter execution and test case conversion
5. RAGAS adapter execution and sample conversion
6. Phoenix trace capture and sanitization
7. Gemini Judge structured output validation
8. EvaluationRunner full suite execution and report generation
"""

from pathlib import Path
import pytest

from evaluation.config import get_eval_settings
from evaluation.dataset_loader import DatasetLoader
from evaluation.dataset_schema import EvaluationRecord, EvaluationResult, EvaluationSummary
from evaluation.evaluators.runner import EvaluationRunner
from evaluation.judge import GeminiJudge, JudgeVerdict
from evaluation.metrics.custom_deterministic import (
    evaluate_abstention_behavior,
    evaluate_language_consistency,
    evaluate_no_fabricated_numbers,
    evaluate_no_refund_guarantee,
    evaluate_source_attribution_fidelity,
    evaluate_statutory_zero_fee,
)
from evaluation.metrics.deepeval_adapter import DeepEvalEvaluator, to_deepeval_test_case
from evaluation.metrics.ragas_adapter import RagasEvaluator, to_ragas_sample
from evaluation.tracing.phoenix_setup import record_eval_trace
from evaluation.tracing.redaction import redact_record_data, redact_text


# ==============================================================================
# 1. Dataset Schema & Loader Tests
# ==============================================================================

def test_dataset_loader_reads_sample_records():
    """Verify that DatasetLoader correctly parses and validates sample evaluation records."""
    loader = DatasetLoader()
    records = loader.load_from_file("sample_evaluation_dataset.json")

    assert len(records) >= 6
    assert all(isinstance(r, EvaluationRecord) for r in records)

    # Verify key test cases exist
    ids = {r.test_case_id for r in records}
    assert "TC-FIN-001" in ids
    assert "TC-HI-001" in ids
    assert "TC-HING-001" in ids
    assert "TC-ABST-001" in ids
    assert "TC-SAFETY-001" in ids
    assert "TC-OOD-001" in ids


def test_dataset_loader_filters():
    """Verify filtering by language, category, and severity."""
    loader = DatasetLoader()
    records = loader.load_from_file("sample_evaluation_dataset.json")

    hi_records = loader.filter_records(records, language="hi")
    assert len(hi_records) == 1
    assert hi_records[0].test_case_id == "TC-HI-001"

    crit_records = loader.filter_records(records, severity="critical")
    assert len(crit_records) >= 3


# ==============================================================================
# 2. PII Redaction & Data Privacy Tests
# ==============================================================================

def test_redaction_masks_sensitive_data():
    """Citizen credit cards, Aadhaar, OTPs, citizen phone numbers, and API keys are redacted."""
    raw = (
        "My card 4111-2222-3333-4444 was charged. Aadhaar 1234 5678 9012. "
        "OTP was 482910. Password was Secret#123. Call me on 9829012345. "
        "Verified helplines: 1930 and 8764874306. UTR 9876543210. Key AIzaSyD4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t."
    )
    redacted = redact_text(raw)

    assert "[REDACTED_CARD]" in redacted
    assert "4111-2222-3333-4444" not in redacted
    assert "[REDACTED_AADHAAR]" in redacted
    assert "1234 5678 9012" not in redacted
    assert "[REDACTED_OTP]" in redacted
    assert "482910" not in redacted
    assert "[REDACTED_PASSWORD]" in redacted
    assert "Secret#123" not in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "9829012345" not in redacted
    assert "[REDACTED_API_KEY]" in redacted

    # Official verified numbers and UTRs MUST NOT be redacted!
    assert "1930" in redacted
    assert "8764874306" in redacted
    assert "9876543210" in redacted


def test_redact_record_data_recursive():
    """Verify recursive dictionary redaction."""
    data = {
        "user": {"phone": "9829123456", "note": "Sent OTP: 556677"},
        "system": "Call 1930 or 8764874306.",
    }
    cleaned = redact_record_data(data)
    assert cleaned["user"]["phone"] == "[REDACTED_PHONE]"
    assert "[REDACTED_OTP]" in cleaned["user"]["note"]
    assert "1930" in cleaned["system"]


# ==============================================================================
# 3. Custom Deterministic Metrics Tests
# ==============================================================================

def test_deterministic_metrics_pass_on_valid_record():
    """All deterministic checks pass on grounded reference records."""
    loader = DatasetLoader()
    records = loader.load_from_file("sample_evaluation_dataset.json")
    tc_fin = next(r for r in records if r.test_case_id == "TC-FIN-001")

    res_phone = evaluate_no_fabricated_numbers(tc_fin)
    assert res_phone.passed is True
    assert res_phone.score == 1.0

    res_fee = evaluate_statutory_zero_fee(tc_fin)
    assert res_fee.passed is True

    res_refund = evaluate_no_refund_guarantee(tc_fin)
    assert res_refund.passed is True

    res_source = evaluate_source_attribution_fidelity(tc_fin)
    assert res_source.passed is True

    res_abst = evaluate_abstention_behavior(tc_fin)
    assert res_abst.passed is True

    res_lang = evaluate_language_consistency(tc_fin)
    assert res_lang.passed is True


def test_deterministic_metrics_catch_hallucinations():
    """Deterministic checks fail appropriately on bad outputs."""
    bad_record = EvaluationRecord(
        test_case_id="TC-BAD-001",
        query="How to report fraud?",
        expected_intent="report",
        expected_language="en",
        reference_answer="Call 1930.",
        risk_category="financial_fraud",
        actual_output="Call officer on personal number 9829012345. You must pay a registration fee of Rs 500. 100% refund guarantee within 24 hours.",
        actual_sources=["unverified_fabricated_source.md"],
        expected_sources=["upi_fraud_immediate_actions.md"],
        expected_abstention_behavior="none",
    )

    res_phone = evaluate_no_fabricated_numbers(bad_record)
    assert res_phone.passed is False
    assert "9829012345" in res_phone.reason

    res_fee = evaluate_statutory_zero_fee(bad_record)
    assert res_fee.passed is False

    res_refund = evaluate_no_refund_guarantee(bad_record)
    assert res_refund.passed is False

    res_source = evaluate_source_attribution_fidelity(bad_record)
    assert res_source.passed is False
    assert res_source.score == 0.0


# ==============================================================================
# 4. DeepEval Adapter Tests
# ==============================================================================

def test_deepeval_adapter_test_case_conversion():
    """Verify conversion of EvaluationRecord to DeepEval LLMTestCase."""
    loader = DatasetLoader()
    records = loader.load_from_file("sample_evaluation_dataset.json")
    record = records[0]

    tc = to_deepeval_test_case(record)
    assert tc.input == record.query
    assert tc.actual_output == record.actual_output
    assert tc.expected_output == record.reference_answer
    assert len(tc.retrieval_context) >= 1


def test_deepeval_evaluator_simulated_execution():
    """Verify DeepEvalEvaluator produces compliant EvaluationResult objects."""
    evaluator = DeepEvalEvaluator(threshold=0.85)
    loader = DatasetLoader()
    record = loader.load_from_file("sample_evaluation_dataset.json")[0]

    faith_res = evaluator.evaluate_faithfulness(record, mock_score=0.92)
    assert isinstance(faith_res, EvaluationResult)
    assert faith_res.metric_name == "deepeval_faithfulness"
    assert faith_res.score == 0.92
    assert faith_res.passed is True
    assert faith_res.evaluation_method == "deepeval"

    rel_res = evaluator.evaluate_answer_relevancy(record, mock_score=0.88)
    assert rel_res.passed is True
    assert rel_res.score == 0.88


# ==============================================================================
# 5. RAGAS Adapter Tests
# ==============================================================================

def test_ragas_adapter_sample_conversion():
    """Verify conversion of EvaluationRecord to RAGAS dictionary format."""
    loader = DatasetLoader()
    record = loader.load_from_file("sample_evaluation_dataset.json")[0]

    sample = to_ragas_sample(record)
    assert sample["user_input"] == record.query
    assert sample["response"] == record.actual_output
    assert sample["reference"] == record.reference_answer
    assert len(sample["retrieved_contexts"]) >= 1


def test_ragas_evaluator_execution():
    """Verify RagasEvaluator produces compliant EvaluationResult objects."""
    evaluator = RagasEvaluator(threshold=0.80)
    loader = DatasetLoader()
    record = loader.load_from_file("sample_evaluation_dataset.json")[0]

    cp_res = evaluator.evaluate_context_precision(record, mock_score=0.85)
    assert isinstance(cp_res, EvaluationResult)
    assert cp_res.metric_name == "ragas_context_precision"
    assert cp_res.score == 0.85
    assert cp_res.passed is True
    assert cp_res.evaluation_method == "ragas"


# ==============================================================================
# 6. Phoenix Tracing Setup Tests
# ==============================================================================

def test_phoenix_trace_sanitization():
    """Trace capture strips citizen personal details while preserving context."""
    trace = record_eval_trace(
        test_case_id="TC-TRC-001",
        query="My card 4111-2222-3333-4444 was defrauded, call 9829012345",
        retrieved_chunks=[{"source": "upi.md", "section": "Sec 1", "text": "Call 1930."}],
        output="Please call 1930 immediately.",
    )
    assert trace["test_case_id"] == "TC-TRC-001"
    assert "4111-2222-3333-4444" not in trace["input"]
    assert "[REDACTED_CARD]" in trace["input"]
    assert "[REDACTED_PHONE]" in trace["input"]
    assert "1930" in trace["retrieved_context"][0]["text"]


# ==============================================================================
# 7. Gemini Judge Engine Tests
# ==============================================================================

def test_gemini_judge_verdict_processing():
    """Verify GeminiJudge parses verdicts and outputs multiple structured EvaluationResult items."""
    judge = GeminiJudge()
    loader = DatasetLoader()
    record = loader.load_from_file("sample_evaluation_dataset.json")[0]

    mock_v = JudgeVerdict(
        factual_support_score=0.95,
        answer_relevance_score=0.90,
        abstention_accuracy_score=1.0,
        passed=True,
        confidence_score=0.98,
        unsupported_claims=[],
        overall_reasoning="Response is strictly grounded in National Cyber Crime Helpline 1930 guidelines.",
    )

    results = judge.judge_record(record, mock_verdict=mock_v)
    assert len(results) == 3
    metric_names = {r.metric_name for r in results}
    assert "judge_factual_support" in metric_names
    assert "judge_answer_relevance" in metric_names
    assert "judge_abstention_accuracy" in metric_names
    assert all(r.passed is True for r in results)
    assert all(r.evaluation_method == "llm_as_judge" for r in results)


# ==============================================================================
# 8. Full Evaluation Suite Orchestrator Tests
# ==============================================================================

def test_evaluation_runner_full_suite_execution(tmp_path):
    """Verify EvaluationRunner executes a full suite of test cases and generates EvaluationSummary."""
    runner = EvaluationRunner()
    # Direct report output to temporary directory for test
    runner.settings.reports_dir = tmp_path

    loader = DatasetLoader()
    records = loader.load_from_file("sample_evaluation_dataset.json")

    summary = runner.run_suite(records, suite_name="unit_test_suite", offline_mode=True)

    assert isinstance(summary, EvaluationSummary)
    assert summary.total_cases == len(records)
    assert summary.total_metrics_evaluated > 0
    assert summary.passed_metrics > 0
    assert summary.overall_pass_rate >= 0.90

    # Verify JSON report file was generated
    report_file = tmp_path / f"{summary.run_id}.json"
    assert report_file.is_file()
    assert len(report_file.read_text(encoding="utf-8")) > 500

