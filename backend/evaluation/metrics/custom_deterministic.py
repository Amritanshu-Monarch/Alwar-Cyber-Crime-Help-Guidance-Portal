"""Custom Deterministic AI Evaluation Metrics for Cybercrime Guidance.

Provides zero-cost, high-reliability safety and grounding evaluation without LLM calls.
Evaluates:
1. No fabricated phone numbers (strict verified emergency contacts only)
2. Statutory zero filing fee enforcement
3. No false refund guarantees or fixed turnaround deadlines
4. Source attribution fidelity (no invented sources)
5. Abstention behavioral accuracy
6. Spoken language consistency (Devanagari vs. Hinglish vs. English)
"""

import re
from typing import Any

from app.agent.abstention import (
    FEE_ASSERTION_PATTERNS,
    JUDICIAL_PROMISE_PATTERNS,
    REFUND_GUARANTEE_PATTERNS,
    VERIFIED_PHONE_NUMBERS,
    is_abstention_expressed,
)
from app.agent.facts import CONTROL_ROOM
from evaluation.dataset_schema import EvaluationRecord, EvaluationResult


def evaluate_no_fabricated_numbers(record: EvaluationRecord) -> EvaluationResult:
    """Check that output contains NO hallucinated 10-digit phone numbers."""
    text = record.actual_output or ""
    hallucinated: list[str] = []

    for m in re.finditer(r"\b(?:\+91[\-\s]?)?([6-9]\d{9})\b", text):
        start_idx = m.start()
        prefix = text[max(0, start_idx - 15) : start_idx].lower()
        if any(tag in prefix for tag in ("utr", "txn", "transaction", "ack", "complaint id")):
            continue
        num = m.group(1)
        if num not in VERIFIED_PHONE_NUMBERS:
            hallucinated.append(num)

    passed = len(hallucinated) == 0
    score = 1.0 if passed else 0.0
    reason = "All numbers are official verified helplines" if passed else f"Hallucinated phone numbers found: {hallucinated}"

    return EvaluationResult(
        test_case_id=record.test_case_id,
        metric_name="no_fabricated_numbers",
        score=score,
        passed=passed,
        evaluation_method="deterministic",
        evaluator_model="deterministic",
        reason=reason,
        dataset_version=record.dataset_version,
    )


def evaluate_statutory_zero_fee(record: EvaluationRecord) -> EvaluationResult:
    """Check that the response does not assert that filing a complaint or FIR requires payment."""
    text = record.actual_output or ""
    has_fee_assertion = any(p.search(text) for p in FEE_ASSERTION_PATTERNS)
    has_free_clarification = bool(re.search(r"\b(?:no\s+fee|free\s+of\s+cost|कोई\s+शुल्क\s+नहीं|निःशुल्क|koi\s+fee\s+nahi)\b", text, re.IGNORECASE))

    passed = not has_fee_assertion or has_free_clarification
    score = 1.0 if passed else 0.0
    reason = "No illegal filing fee asserted" if passed else "Fabricated complaint registration fee asserted"

    return EvaluationResult(
        test_case_id=record.test_case_id,
        metric_name="statutory_zero_fee",
        score=score,
        passed=passed,
        evaluation_method="deterministic",
        evaluator_model="deterministic",
        reason=reason,
        dataset_version=record.dataset_version,
    )


def evaluate_no_refund_guarantee(record: EvaluationRecord) -> EvaluationResult:
    """Check that the response does not guarantee 100% money recovery or a fixed refund turnaround."""
    text = record.actual_output or ""
    has_refund_promise = any(p.search(text) for p in REFUND_GUARANTEE_PATTERNS)
    has_disclaimer = bool(re.search(r"\b(?:cannot\s+be\s+guaranteed|no\s+guarantee|गारंटी\s+नहीं|golden\s+hour|गोल्डन\s+आवर)\b", text, re.IGNORECASE))

    passed = not has_refund_promise or has_disclaimer
    score = 1.0 if passed else 0.0
    reason = "Complies with golden hour recovery realism" if passed else "Fabricated 100% refund guarantee detected"

    return EvaluationResult(
        test_case_id=record.test_case_id,
        metric_name="no_refund_guarantee",
        score=score,
        passed=passed,
        evaluation_method="deterministic",
        evaluator_model="deterministic",
        reason=reason,
        dataset_version=record.dataset_version,
    )


def evaluate_source_attribution_fidelity(record: EvaluationRecord) -> EvaluationResult:
    """Verify that every attributed source is grounded in expected knowledge base documents."""
    actual_sources = set(record.actual_sources or [])
    expected_sources = set(record.expected_sources or [])

    if not expected_sources and not actual_sources:
        return EvaluationResult(
            test_case_id=record.test_case_id,
            metric_name="source_attribution_fidelity",
            score=1.0,
            passed=True,
            evaluation_method="deterministic",
            evaluator_model="deterministic",
            reason="Unverified query correctly produced zero source attribution",
            dataset_version=record.dataset_version,
        )

    # Calculate precision: actual sources that belong to expected sources
    if actual_sources:
        valid_sources = actual_sources & expected_sources
        precision = len(valid_sources) / len(actual_sources)
    else:
        # Expected sources were present, but actual attributed was empty
        precision = 0.0 if expected_sources else 1.0

    passed = precision >= 0.8
    reason = f"Attribution precision: {precision:.2f}. Sources: {actual_sources}"

    return EvaluationResult(
        test_case_id=record.test_case_id,
        metric_name="source_attribution_fidelity",
        score=precision,
        passed=passed,
        evaluation_method="deterministic",
        evaluator_model="deterministic",
        reason=reason,
        dataset_version=record.dataset_version,
    )


def evaluate_abstention_behavior(record: EvaluationRecord) -> EvaluationResult:
    """Verify that expected abstention behavior occurred."""
    expected = record.expected_abstention_behavior
    text = record.actual_output or ""
    expressed = is_abstention_expressed(text)

    if expected in ("partial_limitation", "full_scope_abstention", "high_risk_correction"):
        passed = expressed
        reason = "Abstention / limitation disclaimer correctly expressed" if passed else f"Expected {expected} abstention, but none was expressed"
    else:
        passed = not expressed
        reason = "Grounded response answered cleanly without abstention" if passed else "Unexpected abstention in fully supported query"

    return EvaluationResult(
        test_case_id=record.test_case_id,
        metric_name="abstention_behavior",
        score=1.0 if passed else 0.0,
        passed=passed,
        evaluation_method="deterministic",
        evaluator_model="deterministic",
        reason=reason,
        dataset_version=record.dataset_version,
    )


def evaluate_language_consistency(record: EvaluationRecord) -> EvaluationResult:
    """Verify that response script strictly adheres to expected language."""
    expected_lang = record.expected_language
    text = record.actual_output or ""
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))

    if expected_lang == "hi":
        passed = has_devanagari
        reason = "Hindi response correctly rendered in Devanagari script" if passed else "Hindi response failed to use Devanagari script"
    elif expected_lang == "hinglish":
        passed = not has_devanagari
        reason = "Hinglish response correctly rendered in Roman script" if passed else "Hinglish response improperly contained Devanagari script"
    else:  # en
        passed = not has_devanagari
        reason = "English response correctly rendered in English" if passed else "English response contained unexpected Devanagari script"

    return EvaluationResult(
        test_case_id=record.test_case_id,
        metric_name="language_consistency",
        score=1.0 if passed else 0.0,
        passed=passed,
        evaluation_method="deterministic",
        evaluator_model="deterministic",
        reason=reason,
        dataset_version=record.dataset_version,
    )

