"""Configuration settings for AI evaluation infrastructure."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    """Evaluation framework configuration."""
    model_config = SettingsConfigDict(env_prefix="EVAL_", env_file=".env", extra="ignore")

    # Evaluator LLM (Uses existing Gemini API integration)
    judge_model: str = Field(default="gemini-2.5-flash", description="Model used for LLM-as-a-judge")
    judge_fallback_model: str = Field(default="gemini-2.0-flash", description="Fallback judge model")
    judge_temperature: float = Field(default=0.0, description="Deterministic temperature for evaluation")
    judge_timeout_s: float = Field(default=20.0, description="Judge call timeout in seconds")

    # Evaluation Paths
    eval_dir: Path = Field(default=Path(__file__).resolve().parent)
    datasets_dir: Path = Field(default=Path(__file__).resolve().parent / "datasets")
    reports_dir: Path = Field(default=Path(__file__).resolve().parent / "reports")

    # Framework Toggles (Allows isolated execution without mandatory cloud dependencies)
    enable_deepeval: bool = Field(default=True)
    enable_ragas: bool = Field(default=True)
    enable_phoenix: bool = Field(default=True)

    # Phoenix Tracing Configuration
    phoenix_host: str = Field(default="127.0.0.1")
    phoenix_port: int = Field(default=6006)
    phoenix_project_name: str = Field(default="alwar-cyber-eval")

    # Safety & Threshold Defaults
    faithfulness_threshold: float = Field(default=0.85)
    relevancy_threshold: float = Field(default=0.80)
    context_precision_threshold: float = Field(default=0.75)


def get_eval_settings() -> EvalSettings:
    return EvalSettings()

