from pathlib import Path

import pytest

from app.config import get_settings
from app.services.rag import Chunk, KnowledgeBase


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="yaml-record-summary-0",
            text="YAML record summary",
            source="record.yaml",
            section="Summary",
            domain="financial_fraud",
            subtopic="record",
            jurisdiction="national",
            authority="test",
        ),
        Chunk(
            id="markdown-record-0",
            text="Markdown record",
            source="record.md",
            section="Guide",
            domain="reporting",
            subtopic="reporting",
            jurisdiction="national",
            authority="test",
        ),
    ]


def _kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KnowledgeBase:
    settings = get_settings().model_copy(update={"chroma_dir": tmp_path / "chroma"})
    kb = KnowledgeBase(settings, client=None)
    chunks = _chunks()
    monkeypatch.setattr(kb, "_kb_chunks", lambda: chunks)
    monkeypatch.setattr(
        kb,
        "_embed",
        lambda texts, task_type: [[float(i + 1), 0.0] for i, _ in enumerate(texts)],
    )
    return kb


def test_empty_index_ingests_all_chunks_and_metadata(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)

    assert kb.ingest() == 2
    integrity = kb.verify_index_integrity()

    assert integrity.healthy
    assert integrity.expected_count == 2
    assert integrity.indexed_count == 2
    assert integrity.source_coverage["record.yaml"].missing == 0
    record = kb._col.get(ids=["yaml-record-summary-0"], include=["metadatas"])
    assert record["metadatas"][0]["source"] == "record.yaml"
    assert record["metadatas"][0]["domain"] == "financial_fraud"


def test_complete_index_is_healthy_and_repeated_ingest_is_idempotent(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        kb,
        "_embed",
        lambda texts, task_type: (
            calls.append(len(texts)) or [[float(i + 1), 0.0] for i, _ in enumerate(texts)]
        ),
    )

    kb.ingest()
    marker = (kb._s.chroma_dir / "kb_v2.sha256").read_text(encoding="utf-8")
    assert not kb.ensure_fresh()
    assert kb.ingest() == 2
    assert calls == [2]
    assert (kb._s.chroma_dir / "kb_v2.sha256").read_text(encoding="utf-8") == marker


@pytest.mark.parametrize("missing_id", ["yaml-record-summary-0", "markdown-record-0"])
def test_missing_source_chunk_is_detected_and_repaired(tmp_path, monkeypatch, missing_id):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    kb._col.delete(ids=[missing_id])

    integrity = kb.verify_index_integrity()
    assert missing_id in integrity.missing_ids
    assert not integrity.healthy
    assert kb.ensure_fresh()
    assert kb.verify_index_integrity().healthy


def test_matching_fingerprint_cannot_hide_missing_ids(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    kb._col.delete(ids=["yaml-record-summary-0"])
    (kb._s.chroma_dir / "kb_v2.sha256").write_text(
        kb._fingerprint(), encoding="utf-8"
    )

    assert kb.ensure_fresh()
    assert kb.verify_index_integrity().healthy


def test_correct_count_with_wrong_ids_is_detected(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    kb._col.delete(ids=["yaml-record-summary-0"])
    kb._col.upsert(
        ids=["unexpected-id"],
        documents=["unexpected"],
        metadatas=[{"source": "obsolete.md"}],
        embeddings=[[3.0, 0.0]],
    )

    integrity = kb.verify_index_integrity()
    assert integrity.missing_ids == ("yaml-record-summary-0",)
    assert integrity.unexpected_ids == ("unexpected-id",)
    assert not integrity.healthy
    kb.ingest()
    assert kb.verify_index_integrity().healthy


def test_source_coverage_detects_entirely_missing_source(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    kb._col.delete(ids=["yaml-record-summary-0"])

    coverage = kb.verify_index_integrity().source_coverage["record.yaml"]
    assert coverage.expected == 1
    assert coverage.indexed == 0
    assert coverage.missing == 1


def test_metadata_mismatch_is_detected(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    kb._col.upsert(
        ids=["yaml-record-summary-0"],
        documents=["YAML record summary"],
        metadatas=[{
            "source": "record.yaml",
            "section": "Wrong section",
            "domain": "financial_fraud",
            "subtopic": "record",
            "jurisdiction": "national",
            "authority": "test",
        }],
        embeddings=[[1.0, 0.0]],
    )

    integrity = kb.verify_index_integrity()
    assert "yaml-record-summary-0:section" in integrity.metadata_errors
    assert not integrity.healthy


def test_duplicate_expected_ids_are_detected(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    duplicate = _chunks() + [_chunks()[0]]
    monkeypatch.setattr(kb, "_kb_chunks", lambda: duplicate)

    integrity = kb.verify_index_integrity()
    assert integrity.duplicate_expected_ids == ("yaml-record-summary-0",)
    assert not integrity.healthy


def test_partial_embedding_failure_does_not_update_marker(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    marker_path = kb._s.chroma_dir / "kb_v2.sha256"
    previous_marker = marker_path.read_text(encoding="utf-8")
    expanded = _chunks() + [
        Chunk(
            id="new-yaml-record-0",
            text="New YAML record",
            source="new.yaml",
            section="Summary",
            domain="general",
        )
    ]
    monkeypatch.setattr(kb, "_kb_chunks", lambda: expanded)
    monkeypatch.setattr(
        kb, "_embed", lambda texts, task_type: (_ for _ in ()).throw(RuntimeError("quota"))
    )

    with pytest.raises(RuntimeError, match="Embedding generation failed"):
        kb.ingest()
    assert marker_path.read_text(encoding="utf-8") == previous_marker
    assert "new-yaml-record-0" in kb.verify_index_integrity().missing_ids


def test_failed_upsert_does_not_update_marker(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    marker_path = kb._s.chroma_dir / "kb_v2.sha256"
    previous_marker = marker_path.read_text(encoding="utf-8")
    kb._col.delete(ids=["yaml-record-summary-0"])
    monkeypatch.setattr(
        kb._col,
        "upsert",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )

    with pytest.raises(RuntimeError, match="Chroma upsert failed"):
        kb.ingest()
    assert marker_path.read_text(encoding="utf-8") == previous_marker


def test_changed_content_updates_existing_chunk(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    changed = _chunks()
    changed[0] = Chunk(
        id=changed[0].id,
        text="Changed YAML record summary",
        source=changed[0].source,
        section=changed[0].section,
        domain=changed[0].domain,
        subtopic=changed[0].subtopic,
        jurisdiction=changed[0].jurisdiction,
        authority=changed[0].authority,
    )
    monkeypatch.setattr(kb, "_kb_chunks", lambda: changed)

    kb.ingest()
    record = kb._col.get(ids=["yaml-record-summary-0"], include=["documents"])
    assert record["documents"] == ["Changed YAML record summary"]
    assert kb.verify_index_integrity().healthy


def test_removed_source_chunk_is_deleted_only_after_reconciliation(tmp_path, monkeypatch):
    kb = _kb(tmp_path, monkeypatch)
    kb.ingest()
    monkeypatch.setattr(kb, "_kb_chunks", lambda: _chunks()[:1])

    kb.ingest()
    integrity = kb.verify_index_integrity()
    assert integrity.healthy
    assert integrity.unexpected_ids == ()
    assert kb._col.count() == 1
