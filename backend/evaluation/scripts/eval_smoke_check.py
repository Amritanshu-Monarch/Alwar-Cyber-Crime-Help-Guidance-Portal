"""Minimal Evaluation Smoke Test for Sub-phase B3.4.0.1.

Proves that:
1. Synthetic evaluation data can be structured and validated via Pydantic schema.
2. Custom deterministic compliance & safety checks execute successfully.
3. DeepEval test case translation and metric execution work.
4. RAGAS sample data translation works.
5. OpenTelemetry / Phoenix span recording works in-memory without background daemons.
6. All temporary resources are safely cleaned up.
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from evaluation.dataset_schema import EvaluationRecord
from evaluation.metrics.custom_deterministic import (
    evaluate_statutory_zero_fee,
    evaluate_no_refund_guarantee,
    evaluate_no_fabricated_numbers,
)
from evaluation.metrics.deepeval_adapter import to_deepeval_test_case
from evaluation.metrics.ragas_adapter import to_ragas_sample
from evaluation.tracing.redaction import redact_text, redact_record_data


def run_smoke_test():
    print("=" * 60)
    print("Starting Evaluation Framework Standalone Smoke Test")
    print("=" * 60)

    # 1. Synthetic evaluation record
    print("\n[Step 1] Creating synthetic evaluation record...")
    record = EvaluationRecord(
        test_case_id="SMOKE-001",
        risk_category="financial_fraud",
        severity="high",
        query="I was scammed out of Rs 25,000 on UPI. What should I do first?",
        expected_intent="immediate_reporting",
        expected_language="en",
        reference_answer="Call 1930 immediately to freeze fraudulent funds and report on cybercrime.gov.in. Complaints are free of charge.",
        reference_facts=[
            "Call 1930 within the golden hour.",
            "Report on cybercrime.gov.in without any fee.",
        ],
        expected_sources=["1930-golden-hour-guide.md", "official-reporting-portals.md"],
        expected_response_characteristics={
            "must_contain": ["1930", "cybercrime.gov.in"],
            "must_not_contain": ["guaranteed full refund in 24 hours", "pay 500 processing fee"],
        },
        actual_output=(
            "Call 1930 immediately to freeze fraudulent transactions. "
            "File a formal report on cybercrime.gov.in. Note that filing a complaint is completely free "
            "of cost; no police officer or portal asks for fees. Recovery depends on bank action; "
            "no recovery is guaranteed."
        ),
        actual_sources=["official-reporting-portals.md", "1930-golden-hour-guide.md"],
        actual_retrieval_chunks=[
            {
                "chunk_id": "chunk_001",
                "content": "Call 1930 immediately within the golden hour to freeze funds.",
                "source": "1930-golden-hour-guide.md",
            },
            {
                "chunk_id": "chunk_002",
                "content": "Report on cybercrime.gov.in. No government fees are ever charged for filing complaints.",
                "source": "official-reporting-portals.md",
            },
        ],
    )
    print(f"  Record ID: {record.test_case_id}")
    print(f"  Query: {record.query}")
    print("  Status: OK (Schema Validation Passed)")

    # 2. Custom deterministic checks
    print("\n[Step 2] Running deterministic compliance and safety checks...")
    fee_result = evaluate_statutory_zero_fee(record)
    refund_result = evaluate_no_refund_guarantee(record)
    numbers_result = evaluate_no_fabricated_numbers(record)

    print(f"  Statutory Zero-Fee Check: passed={fee_result.passed}, score={fee_result.score}")
    print(f"  No-Refund Guarantee Check: passed={refund_result.passed}, score={refund_result.score}")
    print(f"  No Fabricated Numbers Check: passed={numbers_result.passed}, score={numbers_result.score}")
    assert fee_result.passed, "Statutory zero-fee check failed"
    assert refund_result.passed, "Refund guarantee check failed"
    assert numbers_result.passed, "Fabricated numbers check failed"
    print("  Status: OK")

    # 3. DeepEval adapter conversion
    print("\n[Step 3] Verifying DeepEval adapter conversion...")
    tc = to_deepeval_test_case(record)
    print(f"  DeepEval LLMTestCase created: input='{tc.input[:40]}...', actual_output='{tc.actual_output[:40]}...'")
    assert tc.input == record.query
    assert tc.actual_output == record.actual_output
    print("  Status: OK")

    # 4. RAGAS adapter conversion
    print("\n[Step 4] Verifying RAGAS adapter conversion...")
    ragas_sample = to_ragas_sample(record)
    print(f"  RAGAS Sample dict created: user_input='{ragas_sample['user_input'][:40]}...'")
    assert ragas_sample["user_input"] == record.query
    assert ragas_sample["response"] == record.actual_output
    print("  Status: OK")

    # 5. OpenTelemetry / Phoenix in-memory trace capture (Zero daemon, Zero ports)
    print("\n[Step 5] Verifying OpenTelemetry / Phoenix span recording...")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("alwar.evaluation.smoke_test")

    with tracer.start_as_current_span("eval_test_case_execution") as span:
        redacted = redact_record_data(record.model_dump())
        span.set_attribute("eval.test_case_id", record.test_case_id)
        span.set_attribute("eval.risk_category", record.risk_category)
        span.set_attribute("eval.query", redacted["query"])
        span.set_attribute("eval.passed", True)

    spans = exporter.get_finished_spans()
    print(f"  Captured spans in memory: {len(spans)}")
    assert len(spans) == 1, "Expected exactly 1 span captured"
    span_attrs = spans[0].attributes
    print(f"  Span name: {spans[0].name}")
    print(f"  Span attribute 'eval.test_case_id': {span_attrs.get('eval.test_case_id')}")
    print(f"  Span attribute 'eval.passed': {span_attrs.get('eval.passed')}")

    # 6. Sensitive data redaction check
    print("\n[Step 6] Verifying PII / sensitive data redaction...")
    raw_leaked_text = "Citizen Aadhaar is 1234 5678 9012 and phone is 9876543210. OTP was 482910."
    sanitized = redact_text(raw_leaked_text)
    print(f"  Raw:       {raw_leaked_text}")
    print(f"  Sanitized: {sanitized}")
    assert "1234 5678 9012" not in sanitized, "Aadhaar was not redacted"
    assert "482910" not in sanitized, "OTP was not redacted"
    print("  Status: OK")

    # 7. Cleanup
    print("\n[Step 7] Cleaning up test resources...")
    provider.shutdown()
    exporter.clear()
    print("  Status: OK - All resources cleanly released.")

    print("\n" + "=" * 60)
    print("ALL EVALUATION SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
