"""Standard Evaluation Dataset Schema & Result Models.

Enforces strict typing and validation for reusable evaluation records, test cases,
and multi-framework evaluation results across DeepEval, RAGAS, custom checks, and LLM-as-a-Judge.
"""

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


class EvaluationRecord(BaseModel):
    """Reusable evaluation dataset record covering factual ground truth and expected system behavior."""

    # Unique identifier & Metadata
    test_case_id: str = Field(description="Unique stable ID e.g. TC-FIN-001")
    dataset_version: str = Field(default="1.0.0", description="Dataset semantic version")
    risk_category: str = Field(
        description="Category e.g. financial_fraud, sextortion, deepfake, fee_hallucination, refund_guarantee, out_of_domain"
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        default="medium", description="Risk impact if system fails this test case"
    )
    evaluation_tags: list[str] = Field(default_factory=list, description="Categorical tags e.g. ['hindi', 'p0', 'voice']")

    # Input Specification
    query: str = Field(description="User query string")
    conversation_history: list[dict[str, str]] | None = Field(
        default=None, description="Prior turns as [{'role': 'user'|'assistant', 'content': '...'}]"
    )
    expected_intent: str = Field(description="Primary intent e.g. report_fraud, seek_helpline, out_of_scope")
    expected_language: Literal["en", "hi", "hinglish"] = Field(
        default="en", description="Expected conversational language"
    )

    # Reference Ground Truth (Human / Authoritative Ground Truth)
    reference_answer: str = Field(description="Canonical authoritative reference response")
    reference_facts: list[str] = Field(
        default_factory=list, description="Atomic factual statements that must be supported"
    )
    expected_sources: list[str] = Field(
        default_factory=list, description="Authorized knowledge base filenames expected to support the answer"
    )

    # Expected Behavioral Safeguards
    expected_abstention_behavior: Literal[
        "none", "partial_limitation", "full_scope_abstention", "high_risk_correction"
    ] = Field(
        default="none",
        description="Expected abstention mode when unverified details or out-of-domain queries occur"
    )
    expected_response_characteristics: dict[str, Any] = Field(
        default_factory=dict,
        description="Behavioral constraints: must_contain, must_not_contain, max_sentences"
    )

    # Actual Execution Outputs (Populated during evaluation run)
    actual_output: str | None = Field(default=None, description="Model generated reply")
    actual_sources: list[str] | None = Field(default=None, description="Sources attributed by the system")
    actual_retrieval_chunks: list[dict[str, Any]] | None = Field(
        default=None, description="Retrieved context chunks from RAG pipeline"
    )
    actual_citation_note: str | None = Field(default=None, description="Citation disclaimer attached to response")


class EvaluationResult(BaseModel):
    """Standardized result for an individual metric evaluated against a test case."""

    test_case_id: str
    metric_name: str
    score: float = Field(ge=0.0, le=1.0, description="Normalized metric score between 0.0 and 1.0")
    passed: bool
    evaluation_method: Literal["deterministic", "deepeval", "ragas", "llm_as_judge"]
    evaluator_model: str = Field(default="deterministic", description="Model name or 'deterministic'")
    reason: str = Field(default="", description="Detailed qualitative justification for the score")
    dataset_version: str = Field(default="1.0.0")
    execution_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EvaluationSummary(BaseModel):
    """Aggregated summary of an evaluation suite run."""

    run_id: str
    dataset_version: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_cases: int
    total_metrics_evaluated: int
    passed_metrics: int
    failed_metrics: int
    overall_pass_rate: float
    metric_averages: dict[str, float]
    results: list[EvaluationResult]

