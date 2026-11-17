"""Dataset loader and validator for AI evaluation records."""

import json
from pathlib import Path
from typing import Any
import yaml

from evaluation.config import get_eval_settings
from evaluation.dataset_schema import EvaluationRecord


class DatasetLoader:
    """Loads, validates, and filters evaluation datasets."""

    def __init__(self, datasets_dir: Path | None = None):
        settings = get_eval_settings()
        self.datasets_dir = datasets_dir or settings.datasets_dir

    def load_from_file(self, file_path: str | Path) -> list[EvaluationRecord]:
        path = Path(file_path)
        if not path.is_file():
            # If relative, check datasets_dir
            path = self.datasets_dir / file_path
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation dataset not found: {file_path}")

        raw_data: list[dict[str, Any]]
        if path.suffix.lower() in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or []
        else:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Dataset root must be a list of records, got {type(raw_data).__name__}")

        records: list[EvaluationRecord] = []
        for i, item in enumerate(raw_data):
            try:
                record = EvaluationRecord.model_validate(item)
                records.append(record)
            except Exception as e:
                raise ValueError(f"Record validation failed at index {i} in {path.name}: {e}") from e

        return records

    def filter_records(
        self,
        records: list[EvaluationRecord],
        language: str | None = None,
        risk_category: str | None = None,
        severity: str | None = None,
        tag: str | None = None,
    ) -> list[EvaluationRecord]:
        filtered = records
        if language:
            filtered = [r for r in filtered if r.expected_language == language]
        if risk_category:
            filtered = [r for r in filtered if r.risk_category == risk_category]
        if severity:
            filtered = [r for r in filtered if r.severity == severity]
        if tag:
            filtered = [r for r in filtered if tag in r.evaluation_tags]
        return filtered

    def save_records(self, records: list[EvaluationRecord], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

