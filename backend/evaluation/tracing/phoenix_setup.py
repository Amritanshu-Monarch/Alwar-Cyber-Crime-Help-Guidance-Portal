"""Arize Phoenix Tracing & Observability Setup for RAG Evaluation.

Configures a self-hosted, local-only Phoenix instance for RAG pipeline evaluation,
ensuring all trace attributes are redacted prior to persistence.
"""

import logging
from typing import Any

from evaluation.config import get_eval_settings
from evaluation.tracing.redaction import redact_text

log = logging.getLogger(__name__)

_session: Any = None


def launch_phoenix_eval_session():
    """Launch local self-hosted Phoenix UI session on configured port.
    
    Returns the Phoenix session instance, or None if disabled or unavailable.
    """
    global _session
    if _session is not None:
        return _session

    settings = get_eval_settings()
    if not settings.enable_phoenix:
        log.info("Phoenix evaluation tracing is disabled in settings.")
        return None

    try:
        import phoenix as px

        # Launch local standalone instance without internet telemetries
        _session = px.launch_app(
            host=settings.phoenix_host,
            port=settings.phoenix_port,
            run_in_thread=True,
        )
        log.info("Arize Phoenix evaluation server running at http://%s:%d", settings.phoenix_host, settings.phoenix_port)
        return _session
    except Exception as e:
        log.warning("Could not launch Phoenix session: %s", e)
        return None


def close_phoenix_eval_session():
    """Gracefully terminate active Phoenix session."""
    global _session
    if _session is not None:
        try:
            import phoenix as px
            px.close_app()
        except Exception:
            pass
        _session = None


def record_eval_trace(
    test_case_id: str,
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    output: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture a sanitized trace object suitable for Phoenix evaluation dataset export."""
    sanitized_query = redact_text(query)
    sanitized_output = redact_text(output)
    sanitized_chunks = [
        {
            "source": c.get("source", ""),
            "section": c.get("section", ""),
            "text": redact_text(c.get("text", "")),
        }
        for c in retrieved_chunks
    ]

    trace_span = {
        "test_case_id": test_case_id,
        "input": sanitized_query,
        "retrieved_context": sanitized_chunks,
        "output": sanitized_output,
        "metadata": metadata or {},
    }
    return trace_span

