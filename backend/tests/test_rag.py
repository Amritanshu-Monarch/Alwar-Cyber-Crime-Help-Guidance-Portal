from app.services.rag import chunk, expand_query, parse_chunks, rrf, tokenize

DOC = """# Reporting guide

## First steps
Call 1930 at once.

## Documents
Bank statement and screenshots.

Second paragraph about ID proof.
"""


def test_parse_chunks_prefixes_title_and_section():
    chunks = parse_chunks(DOC, "reporting-guide.md")
    assert [c.section for c in chunks] == ["First steps", "Documents"]
    assert chunks[0].text.startswith("Reporting guide > First steps\n")
    assert all(c.source == "reporting-guide.md" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_parse_chunks_flat_files_use_single_hash_headings():
    chunks = parse_chunks("# Helplines\n\nCall 1930.\n\n# Evidence\n\nKeep chats.\n", "portal.md")
    assert [c.section for c in chunks] == ["Helplines", "Evidence"]


def test_parse_chunks_splits_long_sections():
    long_doc = "# T\n\n## S\n" + "\n\n".join("word " * 60 for _ in range(6))
    chunks = parse_chunks(long_doc, "t.md", size=400)
    assert len(chunks) > 1 and all(c.section == "S" for c in chunks)


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["x", "y"], ["z", "x"]])[0] == "x"
    assert rrf([]) == []


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("How do I report the UPI fraud?") == ["report", "upi", "fraud"]


def test_telegram_hacked_query_expands_to_verified_recovery_terms():
    expanded = expand_query("Mera Telegram account hack ho gaya, recover kaise karun?")
    assert "terminate telegram sessions" in expanded
    assert "two step verification" in expanded
    assert "telegram support" in expanded


def test_chunk_packs_paragraphs_and_respects_size():
    text = "\n\n".join(f"para {i} " + "x" * 200 for i in range(6))
    out = chunk(text, size=500)
    assert len(out) > 1
    assert all(len(c) <= 700 for c in out)
    assert "para 0" in out[0] and "para 5" in out[-1]


def test_chunk_empty():
    assert chunk("") == []


# ==============================================================================
# Sub-phase B2: RAG Retrieval Quality Validation Tests
# ==============================================================================

import pytest
from app.config import get_settings
from app.services.rag import KnowledgeBase
from app.agent.grounding import attribute_evidence


@pytest.fixture(scope="module")
def deterministic_kb():
    """Deterministic, offline KnowledgeBase fixture for fast unit tests without API calls."""
    settings = get_settings()
    # Initializing with client=None ensures 100% deterministic local in-memory BM25 retrieval
    return KnowledgeBase(settings, client=None)


def test_rag_retrieval_english_query(deterministic_kb):
    """Verify that English queries retrieve relevant official documents."""
    hits = deterministic_kb.search("How to report financial cyber fraud on 1930?")
    assert hits, "Expected retrieval hits for English fraud reporting query"
    assert any("1930" in h["text"] for h in hits)
    assert any("fraud" in h["source"].lower() or "reporting" in h["source"].lower() for h in hits)


def test_rag_retrieval_hindi_query(deterministic_kb):
    """Verify that Devanagari Hindi queries retrieve relevant official documents."""
    hits = deterministic_kb.search("मेरे बैंक खाते से फर्जी पैसे कट गए 1930 पर शिकायत कैसे करें")
    assert hits, "Expected retrieval hits for Hindi financial loss query"
    assert any("1930" in h["text"] for h in hits)
    assert any(h["domain"] in ("financial_fraud", "reporting") for h in hits)


def test_rag_retrieval_hinglish_query(deterministic_kb):
    """Verify that colloquial Hinglish queries retrieve relevant official documents."""
    hits = deterministic_kb.search("freeze hua paisa refund kaise hoga mrm portal")
    assert hits, "Expected retrieval hits for Hinglish MRM query"
    sources = [h["source"] for h in hits]
    assert any("mrm_portal_recovery" in s for s in sources)


def test_rag_retrieval_paraphrased_query(deterministic_kb):
    """Verify that paraphrased descriptions retrieve relevant guidance without exact keyword overlap."""
    hits = deterministic_kb.search("someone demanding ransom using fake morphed nude photo")
    assert hits, "Expected retrieval hits for deepfake extortion query"
    domains = [h["domain"] for h in hits]
    assert any(d in ("deepfake", "sextortion", "harassment") for d in domains)
    assert all(h["domain"] != "financial_fraud" for h in hits)


def test_rag_irrelevant_query_suppression(deterministic_kb):
    """Quality gate: reject out-of-domain and non-cybercrime queries."""
    assert deterministic_kb.search("How to bake a chocolate cake?") == []
    assert deterministic_kb.search("tell me a funny joke") == []
    assert deterministic_kb.search("capital of France") == []


def test_rag_yaml_document_retrieval(deterministic_kb):
    """Verify that structured YAML documents in backend/kb/yaml/ are fully indexed and retrievable."""
    hits = deterministic_kb.search("how to disable call forwarding *401*")
    assert hits, "Expected retrieval hits for call forwarding query"
    sources = [h["source"] for h in hits]
    assert any("call_forwarding_scam" in s for s in sources)


def test_rag_markdown_document_retrieval(deterministic_kb):
    """Verify that Markdown documents in backend/kb/ subdirectories are fully indexed and retrievable."""
    hits = deterministic_kb.search("Alwar Kotwali Police Station phone number and SHO")
    assert hits, "Expected retrieval hits for Alwar police station query"
    assert any(h["source"].endswith(".md") for h in hits)


def test_rag_domain_filtering_isolation(deterministic_kb):
    """Metadata domain filtering: strictly isolate deepfake/social media from financial fraud."""
    hits = deterministic_kb.search("Someone created an intimate deepfake video", domains=["deepfake"])
    assert hits, "Expected retrieval hits for deepfake domain"
    assert all(h["domain"] != "financial_fraud" for h in hits)


def test_rag_platform_filtering_isolation_for_telegram_recovery(deterministic_kb):
    """A named platform must not retrieve another platform's recovery procedure."""
    hits = deterministic_kb.search(
        "Mera Telegram account hack ho gaya, recover kaise karun?",
        k=6,
        domains=["account_hacking", "social_media"],
    )
    sources = [h["source"].lower() for h in hits]
    assert any("telegram" in source for source in sources)
    assert not any(
        platform in source
        for source in sources
        for platform in ("facebook", "instagram", "whatsapp", "gmail", "twitter")
    )


def test_rag_empty_and_conversation_queries_safe(deterministic_kb):
    """Empty strings, whitespace, and conversational greetings return empty hits safely."""
    assert deterministic_kb.search("") == []
    assert deterministic_kb.search("   ") == []
    assert deterministic_kb.search("hi") == []
    assert deterministic_kb.search("Namaste") == []


def test_rag_reranker_failure_fallback():
    """Reranker failure resilience: if LLM reranker raises an exception, fall back to deterministic ranker without crashing."""
    settings = get_settings()
    class _FailingClient:
        class models:
            @staticmethod
            def generate_content(*args, **kwargs):
                raise RuntimeError("LLM quota exceeded or service unavailable")

    kb_with_failing_llm = KnowledgeBase(settings, client=_FailingClient())
    # Enable rerank setting to trigger LLM rerank branch
    kb_with_failing_llm._s = settings.model_copy(update={"rerank": True})
    hits = kb_with_failing_llm.search("how to stop call forwarding *401*")
    assert hits, "Expected fallback to deterministic ranker on LLM failure"
    assert any("call_forwarding_scam" in h["source"] for h in hits)


def test_rag_chroma_resilience_bm25_only(monkeypatch):
    """ChromaDB failure resilience: if ChromaDB cannot be initialized, KB operates in BM25-only mode."""
    settings = get_settings()
    # Force chroma collection to None as if Chroma failed or is read-only
    kb_bm25_only = KnowledgeBase(settings, client=None)
    kb_bm25_only._col = None
    assert kb_bm25_only.count() == len(kb_bm25_only._chunks)
    hits = kb_bm25_only.search("StopNCII remove morphed photos intimate image")
    assert hits, "Expected hits in BM25-only fallback mode"
    sources = [h["source"] for h in hits]
    assert any("morphed_photos_stopncii" in s for s in sources)


def test_rag_grounding_compatibility(deterministic_kb):
    """Verify end-to-end compatibility between RAG retrieval hits and Sub-phase B1 source attribution."""
    hits = deterministic_kb.search("freeze hua paisa refund kaise hoga mrm portal")
    assert hits, "Expected hits for MRM portal"
    reply = "Call 1930 to freeze funds and submit an application on the MRM portal for lien removal."
    sources, note = attribute_evidence(reply, hits, ["reporting"], lang="en")
    assert any("mrm_portal_recovery" in s for s in sources)
    assert "Verified from official knowledge base:" in note
