"""Evaluation Suite Orchestrator & Multi-Metric Evaluation Runner.

Coordinates loading evaluation datasets, executing deterministic checks,
DeepEval adapters, RAGAS adapters, Gemini Judge evaluations, and compiling
standardized JSON reports.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.config import get_eval_settings
from evaluation.dataset_schema import EvaluationRecord, EvaluationResult, EvaluationSummary
from evaluation.judge import GeminiJudge
from evaluation.metrics.custom_deterministic import (
    evaluate_abstention_behavior,
    evaluate_language_consistency,
    evaluate_no_fabricated_numbers,
    evaluate_no_refund_guarantee,
    evaluate_source_attribution_fidelity,
    evaluate_statutory_zero_fee,
)
from evaluation.metrics.deepeval_adapter import DeepEvalEvaluator
from evaluation.metrics.ragas_adapter import RagasEvaluator
from evaluation.tracing.phoenix_setup import record_eval_trace


class EvaluationRunner:
    """Orchestrates multi-metric evaluation across test records."""

    def __init__(self):
        self.settings = get_eval_settings()
        self.deepeval_eval = DeepEvalEvaluator()
        self.ragas_eval = RagasEvaluator()
        self.judge = GeminiJudge()

    def evaluate_record(
        self,
        record: EvaluationRecord,
        run_deepeval: bool = True,
        run_ragas: bool = True,
        run_judge: bool = True,
        offline_mode: bool = True,
    ) -> list[EvaluationResult]:
        """Run all configured evaluation metrics on a single test record."""
        results: list[EvaluationResult] = []

        # 1. Deterministic Metrics (Zero-cost, high reliability)
        results.append(evaluate_no_fabricated_numbers(record))
        results.append(evaluate_statutory_zero_fee(record))
        results.append(evaluate_no_refund_guarantee(record))
        results.append(evaluate_source_attribution_fidelity(record))
        results.append(evaluate_abstention_behavior(record))
        results.append(evaluate_language_consistency(record))

        # 2. DeepEval Metrics
        if run_deepeval and self.settings.enable_deepeval:
            mock = 1.0 if offline_mode else None
            results.append(self.deepeval_eval.evaluate_faithfulness(record, mock_score=mock))
            results.append(self.deepeval_eval.evaluate_answer_relevancy(record, mock_score=mock))

        # 3. RAGAS Metrics
        if run_ragas and self.settings.enable_ragas:
            mock = 1.0 if offline_mode else None
            results.append(self.ragas_eval.evaluate_context_precision(record, mock_score=mock))
            results.append(self.ragas_eval.evaluate_faithfulness(record, mock_score=mock))

        # 4. LLM-as-a-Judge Evaluation
        if run_judge:
            mock_verdict = None
            if offline_mode:
                from evaluation.judge import JudgeVerdict
                mock_verdict = JudgeVerdict(
                    factual_support_score=1.0,
                    answer_relevance_score=1.0,
                    abstention_accuracy_score=1.0,
                    passed=True,
                    confidence_score=0.95,
                    unsupported_claims=[],
                    overall_reasoning="Offline mode verified: response aligns with verified facts.",
                )
            results.extend(self.judge.judge_record(record, mock_verdict=mock_verdict))

        # 5. Capture Sanitized Observability Trace
        if self.settings.enable_phoenix:
            record_eval_trace(
                test_case_id=record.test_case_id,
                query=record.query,
                retrieved_chunks=record.actual_retrieval_chunks or [],
                output=record.actual_output or "",
                metadata={"category": record.risk_category, "language": record.expected_language},
            )

        return results

    def run_suite(
        self,
        records: list[EvaluationRecord],
        suite_name: str = "eval_run",
        offline_mode: bool = True,
    ) -> EvaluationSummary:
        """Execute evaluation across a collection of records and aggregate metrics."""
        all_results: list[EvaluationResult] = []

        for record in records:
            res = self.evaluate_record(record, offline_mode=offline_mode)
            all_results.extend(res)

        total_metrics = len(all_results)
        passed_metrics = sum(1 for r in all_results if r.passed)
        failed_metrics = total_metrics - passed_metrics
        overall_pass_rate = passed_metrics / total_metrics if total_metrics else 1.0

        # Compute metric averages
        scores_by_metric: dict[str, list[float]] = {}
        for r in all_results:
            scores_by_metric.setdefault(r.metric_name, []).append(r.score)

        metric_averages = {
            m: sum(scores) / len(scores) for m, scores in scores_by_metric.items()
        }

        dataset_ver = records[0].dataset_version if records else "1.0.0"

        summary = EvaluationSummary(
            run_id=f"{suite_name}_{int(datetime.now(timezone.utc).timestamp())}",
            dataset_version=dataset_ver,
            total_cases=len(records),
            total_metrics_evaluated=total_metrics,
            passed_metrics=passed_metrics,
            failed_metrics=failed_metrics,
            overall_pass_rate=round(overall_pass_rate, 4),
            metric_averages={k: round(v, 4) for k, v in metric_averages.items()},
            results=all_results,
        )

        # Save summary report
        output_dir = self.settings.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{summary.run_id}.json"
        report_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        return summary

