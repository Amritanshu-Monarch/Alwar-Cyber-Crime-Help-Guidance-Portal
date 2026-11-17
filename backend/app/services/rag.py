"""Domain-aware Hybrid RAG over YAML and Markdown knowledge base.

Pipeline:
1. Frontmatter, heading & schema-aware atomic chunks (with domain, subtopic, jurisdiction metadata).
2. Supports both rich YAML schemas (e.g. root/kb/*.yaml) and Markdown documents (backend/kb/**/*.md).
3. Domain classification & Hindi/Hinglish query expansion.
4. Metadata-filtered Vector search (Gemini embeddings in ChromaDB) + Unicode-aware BM25.
5. Reciprocal Rank Fusion (RRF).
6. High-precision deterministic multi-signal reranking (+ optional LLM rerank).
7. Calibrated scope guard preventing out-of-domain hallucinations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    genai_errors = None  # type: ignore
    types = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    import math

    class BM25Okapi:  # type: ignore
        """Pure-Python Okapi BM25 implementation fallback for environments without rank_bm25."""
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
            self.k1 = k1
            self.b = b
            self.epsilon = epsilon
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size else 0.0
            self.doc_freqs: list[dict[str, int]] = []
            self.idf: dict[str, float] = {}
            self.doc_len: list[int] = []
            nd: dict[str, int] = {}
            for document in corpus:
                self.doc_len.append(len(document))
                frequencies: dict[str, int] = {}
                for word in document:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    nd[word] = nd.get(word, 0) + 1

            idf_sum = 0.0
            negative_idfs = []
            for word, freq in nd.items():
                idf_val = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
                self.idf[word] = idf_val
                idf_sum += idf_val
                if idf_val < 0:
                    negative_idfs.append(word)

            average_idf = idf_sum / len(self.idf) if self.idf else 0.0
            eps = self.epsilon * average_idf
            for word in negative_idfs:
                self.idf[word] = eps

        def get_scores(self, query: list[str]) -> list[float]:
            scores = [0.0] * self.corpus_size
            doc_len = self.doc_len
            for q in query:
                score_q = self.idf.get(q, 0.0)
                if score_q <= 0:
                    continue
                for i in range(self.corpus_size):
                    freq = self.doc_freqs[i].get(q, 0)
                    if freq == 0:
                        continue
                    denom = freq + self.k1 * (1 - self.b + self.b * doc_len[i] / self.avgdl)
                    scores[i] += score_q * (freq * (self.k1 + 1)) / denom
            return scores

from app.config import Settings, get_settings
from app.services.query_expander import classify_domains, expand_query, tokenize

log = logging.getLogger(__name__)

EMBED_BATCH = 50
CHUNK_CHARS = 900
VEC_K = 8
BM25_K = 8
RERANK_K = 4
RERANK_CHARS = 600
RRF_K = 60

# Calibrated scope guard: reject when query has weak semantic similarity AND low lexical overlap
OOS_DISTANCE = 0.42
BM25_MIN_SCORE = 7.0

# General cross-domain fallback categories always eligible for supporting context
CROSS_CUTTING_DOMAINS = frozenset(
    {"general", "reporting", "evidence", "local_police", "emergency", "portals", "portals_reporting"}
)

PLATFORM_SOURCE_TERMS = {
    "telegram": ("telegram",),
    "whatsapp": ("whatsapp",),
    "facebook": ("facebook",),
    "instagram": ("instagram",),
    "gmail": ("gmail", "google"),
    "twitter": ("twitter", "x_twitter"),
}

RERANK_PROMPT = """You judge relevance for a cyber crime help knowledge base used by Indian citizens.
Question: {query}

Passages:
{passages}

Return a JSON array of the passage numbers that directly help answer the question, most helpful first
(at most {k}). Return [] if none of them are relevant to the question."""

CATEGORY_TO_DOMAIN = {
    "financial_fraud": "financial_fraud",
    "portals_reporting": "reporting",
    "device_malware": "malware",
    "social_media": "social_media",
    "scam_alerts": "financial_fraud",
    "cyber_hygiene": "cyber_safety",
}

ID_DOMAIN_OVERRIDES = {
    # portals / reporting
    "mrm_portal_recovery": "reporting",
    "grm_portal_grievance": "reporting",
    "evidence_checklist": "evidence",
    "thana_finder_help": "local_police",
    "ncrp_vs_fir": "reporting",
    "bank_account_freeze": "reporting",
    # device / malware / telecom
    "call_forwarding_scam": "telecom",
    "phone_lost_stolen": "device",
    "sim_swap_fraud": "telecom",
    "apk_malware_cleanup": "malware",
    "remote_access_scam": "malware",
    "challan_parcel_scam": "malware",
    # social media & harassment
    "whatsapp_hacked": "account_hacking",
    "facebook_hacked": "account_hacking",
    "instagram_hacked": "account_hacking",
    "gmail_google_hacked": "account_hacking",
    "telegram_hacked": "account_hacking",
    "x_twitter_hacked": "account_hacking",
    "morphed_photos_stopncii": "deepfake",
    "deepfake_voice_scam": "deepfake",
    "sextortion_blackmail": "sextortion",
    "cyber_harassment_stalking": "harassment",
    "loan_app_harassment": "harassment",
    "social_media_grievance_officers": "social_media",
    # phishing / scam alerts / cyber hygiene
    "spot_phishing_quiz": "cyber_safety",
    "govt_subsidy_tax_scam": "phishing",
}


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str
    domain: str = "general"
    subtopic: str = ""
    jurisdiction: str = "national"
    authority: str = "official_government"


@dataclass(frozen=True)
class SourceCoverage:
    expected: int
    indexed: int
    missing: int
    unexpected: int


@dataclass(frozen=True)
class IndexIntegrity:
    healthy: bool
    expected_count: int
    indexed_count: int
    missing_ids: tuple[str, ...]
    unexpected_ids: tuple[str, ...]
    duplicate_expected_ids: tuple[str, ...]
    metadata_errors: tuple[str, ...]
    source_coverage: dict[str, SourceCoverage]


def chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Pack paragraphs into ~size-char pieces."""
    pieces, buf = [], ""
    for para in text.split("\n\n"):
        if buf and len(buf) + len(para) > size:
            pieces.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def _extract_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract YAML frontmatter if present and return metadata dict + body text."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_meta = parts[1].strip()
            body = parts[2].strip()
            meta: dict[str, str] = {}
            for line in raw_meta.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
            return meta, body
    return {}, text


def parse_chunks(markdown: str, source: str, size: int = CHUNK_CHARS) -> list[Chunk]:
    """Split on '## ' sections; every chunk is prefixed 'Title > Section' so it is self-describing.
    Parses frontmatter metadata (domain, subtopic, jurisdiction, authority).
    """
    meta, body_text = _extract_frontmatter(markdown)
    domain = meta.get("domain", "general")
    subtopic = meta.get("subtopic", "")
    jurisdiction = meta.get("jurisdiction", "national")
    authority = meta.get("authority", "official_government")

    title, heading, body = "", "", []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        if heading and "".join(body).strip():
            sections.append((heading, "\n".join(body).strip()))

    # files without '## ' sections use '# ' headings as the sections themselves
    flat = not any(line.startswith("## ") for line in body_text.splitlines())
    for line in body_text.splitlines():
        if line.startswith("## ") or (flat and line.startswith("# ")):
            flush()
            heading, body = line.lstrip("# ").strip(), []
        elif line.startswith("# "):
            title = line.lstrip("# ").strip()
        else:
            body.append(line)
    flush()
    title = title or source.removesuffix(".md").replace("-", " ")

    stem = source.removesuffix(".md")
    chunks = []
    for s_idx, (sec, text) in enumerate(sections):
        for p_idx, piece in enumerate(chunk(text, size)):
            chunks.append(
                Chunk(
                    id=f"{stem}-{s_idx}-{p_idx}",
                    text=f"{title} > {sec}\n{piece}",
                    source=source,
                    section=sec,
                    domain=domain,
                    subtopic=subtopic,
                    jurisdiction=jurisdiction,
                    authority=authority,
                )
            )
    return chunks


def parse_yaml_chunks(data: dict, filename: str, size: int = CHUNK_CHARS) -> list[Chunk]:
    """Parse structured YAML KB document into atomic, self-describing chunks.
    Extracts summary, English steps, Hindi steps, keywords/queries, and official links.
    """
    fid = data.get("id") or Path(filename).stem
    cat = data.get("category", "general")
    domain = ID_DOMAIN_OVERRIDES.get(fid, CATEGORY_TO_DOMAIN.get(cat, "general"))
    subtopic = data.get("subtopic") or fid
    jurisdiction = data.get("jurisdiction", "national")
    authority = data.get("source", "official_government")
    title_en = data.get("title_en", "")
    title_hi = data.get("title_hi", "")
    full_title = f"{title_en} / {title_hi}" if (title_en and title_hi) else (title_en or title_hi or fid)

    chunks: list[Chunk] = []

    # 1. Summary Chunk
    sum_en = data.get("summary_en", "")
    sum_hi = data.get("summary_hi", "")
    urgency = data.get("urgency", "medium")
    kw_en = data.get("keywords_en", [])
    kw_hi = data.get("keywords_hi", [])
    kw_hing = data.get("keywords_hinglish", [])
    samples = data.get("sample_queries", [])

    if sum_en or sum_hi:
        sum_body = f"Urgency: {urgency}\nEnglish: {sum_en}\nहिंदी: {sum_hi}"
        if samples:
            sum_body += f"\nCommon Queries: {'; '.join(samples[:5])}"
        chunks.append(
            Chunk(
                id=f"{fid}-summary-0",
                text=f"{full_title} > Summary & Overview\n{sum_body}",
                source=filename,
                section="Summary & Overview",
                domain=domain,
                subtopic=subtopic,
                jurisdiction=jurisdiction,
                authority=authority,
            )
        )

    # 2. English Action Steps
    ans_en = data.get("answer_en", [])
    if ans_en:
        steps_text = "\n".join(
            f"{i+1}. {s}" if not s.strip().startswith(tuple("0123456789")) else s
            for i, s in enumerate(ans_en)
        )
        for p_idx, piece in enumerate(chunk(steps_text, size)):
            chunks.append(
                Chunk(
                    id=f"{fid}-en-steps-{p_idx}",
                    text=f"{title_en} > Immediate Action & Steps (English)\n{piece}",
                    source=filename,
                    section="Immediate Action & Steps (English)",
                    domain=domain,
                    subtopic=subtopic,
                    jurisdiction=jurisdiction,
                    authority=authority,
                )
            )

    # 3. Hindi Action Steps
    ans_hi = data.get("answer_hi", [])
    if ans_hi:
        steps_text = "\n".join(
            f"{i+1}. {s}" if not s.strip().startswith(tuple("0123456789")) else s
            for i, s in enumerate(ans_hi)
        )
        for p_idx, piece in enumerate(chunk(steps_text, size)):
            chunks.append(
                Chunk(
                    id=f"{fid}-hi-steps-{p_idx}",
                    text=f"{title_hi} > तत्काल कदम और समाधान (Hindi Guide)\n{piece}",
                    source=filename,
                    section="तत्काल कदम और समाधान (Hindi Guide)",
                    domain=domain,
                    subtopic=subtopic,
                    jurisdiction=jurisdiction,
                    authority=authority,
                )
            )

    # 4. Search Keywords, Queries & Variations
    if kw_en or kw_hi or kw_hing or samples:
        parts = [
            f"Urgency: {urgency}",
            f"Summary: {sum_en} | {sum_hi}",
            "National Cyber Helpline: 1930 | Portal: cybercrime.gov.in",
        ]
        if kw_en:
            parts.append("Keywords (EN): " + ", ".join(kw_en))
        if kw_hi:
            parts.append("Keywords (HI): " + ", ".join(kw_hi))
        if kw_hing:
            parts.append("Keywords (Hinglish): " + ", ".join(kw_hing))
        if samples:
            parts.append("Common Queries: " + "; ".join(samples))
        chunks.append(
            Chunk(
                id=f"{fid}-keywords-0",
                text=f"{full_title} > Queries & Keywords\n" + "\n".join(parts),
                source=filename,
                section="Queries & Keywords",
                domain=domain,
                subtopic=subtopic,
                jurisdiction=jurisdiction,
                authority=authority,
            )
        )

    # 5. Official Helplines & Links
    links = data.get("official_links", [])
    if links:
        link_lines = [
            f"- {l.get('name_en', '')} ({l.get('name_hi', '')}): {l.get('url', '')}"
            for l in links
        ]
        chunks.append(
            Chunk(
                id=f"{fid}-links-0",
                text=f"{full_title} > Official Helplines & Links\n" + "\n".join(link_lines),
                source=filename,
                section="Official Helplines & Links",
                domain=domain,
                subtopic=subtopic,
                jurisdiction=jurisdiction,
                authority=authority,
            )
        )

    return chunks


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: merge several best-first id lists into one ordering."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def _filter_platform_chunks(query: str, candidates: list[Chunk]) -> list[Chunk]:
    """Keep a named platform's procedure separate from other platform procedures."""
    query_lower = query.lower()
    mentioned = [
        platform
        for platform, terms in PLATFORM_SOURCE_TERMS.items()
        if any(term in query_lower for term in terms)
    ]
    if len(mentioned) != 1:
        return candidates

    target = mentioned[0]
    target_terms = PLATFORM_SOURCE_TERMS[target]
    target_specific = [
        c for c in candidates
        if any(term in c.source.lower() for term in target_terms)
    ]
    if not target_specific:
        return candidates

    other_terms = {
        term
        for platform, terms in PLATFORM_SOURCE_TERMS.items()
        if platform != target
        for term in terms
    }
    return [
        c for c in candidates
        if any(term in c.source.lower() for term in target_terms)
        or not any(term in c.source.lower() for term in other_terms)
    ]


class KnowledgeBase:
    def __init__(self, settings: Settings, client: genai.Client | None = None):
        self._s = settings
        if chromadb is None:
            log.warning("chromadb is not available in environment; vector search disabled")
            self._col = None
        else:
            try:
                settings.chroma_dir.mkdir(parents=True, exist_ok=True)
                db = chromadb.PersistentClient(path=str(settings.chroma_dir))
                self._col = db.get_or_create_collection("alwar_kb_v2", metadata={"hnsw:space": "cosine"})
            except Exception as e:
                log.warning("ChromaDB initialization failed (%s); operating in BM25-only mode", e)
                self._col = None

        if genai is not None:
            self._client = client or genai.Client(api_key=settings.google_api_key or None)
        else:
            self._client = None

        self._chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._rebuild_index()

    def count(self) -> int:
        if self._col is not None:
            return self._col.count()
        return len(self._chunks)

    def _collection_ids(self, page_size: int = 500) -> set[str]:
        """Read only persisted IDs, using pagination to avoid loading payloads."""
        if self._col is None:
            return set()

        ids: set[str] = set()
        offset = 0
        while True:
            page = self._col.get(limit=page_size, offset=offset, include=[])
            page_ids = page.get("ids") or []
            ids.update(page_ids)
            if len(page_ids) < page_size:
                return ids
            offset += len(page_ids)

    @staticmethod
    def _chunk_metadata(c: Chunk) -> dict[str, str]:
        return {
            "source": c.source,
            "section": c.section,
            "domain": c.domain,
            "subtopic": c.subtopic,
            "jurisdiction": c.jurisdiction,
            "authority": c.authority,
        }

    def verify_index_integrity(self, chunks: list[Chunk] | None = None) -> IndexIntegrity:
        """Compare expected disk chunks with persisted IDs, sources and metadata."""
        expected = chunks if chunks is not None else self._kb_chunks()
        expected_counts = Counter(c.id for c in expected)
        expected_map = {c.id: c for c in expected}
        duplicate_expected_ids = tuple(sorted(cid for cid, n in expected_counts.items() if n > 1))

        if self._col is None:
            return IndexIntegrity(
                healthy=True,
                expected_count=len(expected_map),
                indexed_count=0,
                missing_ids=(),
                unexpected_ids=(),
                duplicate_expected_ids=duplicate_expected_ids,
                metadata_errors=(),
                source_coverage={},
            )

        actual_ids = self._collection_ids()
        missing_ids = tuple(sorted(set(expected_map) - actual_ids))
        unexpected_ids = tuple(sorted(actual_ids - set(expected_map)))

        actual_meta: dict[str, dict] = {}
        if actual_ids:
            data = self._col.get(ids=sorted(actual_ids), include=["metadatas"])
            actual_meta = dict(zip(data.get("ids") or [], data.get("metadatas") or []))

        metadata_errors: list[str] = []
        for cid in sorted(actual_ids & set(expected_map)):
            expected_meta = self._chunk_metadata(expected_map[cid])
            actual = actual_meta.get(cid) or {}
            for key, value in expected_meta.items():
                if actual.get(key) != value:
                    metadata_errors.append(f"{cid}:{key}")

        expected_by_source = Counter(c.source for c in expected)
        indexed_by_source = Counter(
            (actual_meta.get(cid) or {}).get("source", "<missing>")
            for cid in actual_ids
        )
        source_coverage: dict[str, SourceCoverage] = {}
        for source in sorted(set(expected_by_source) | set(indexed_by_source)):
            expected_count = expected_by_source.get(source, 0)
            indexed_count = indexed_by_source.get(source, 0)
            source_coverage[source] = SourceCoverage(
                expected=expected_count,
                indexed=indexed_count,
                missing=max(expected_count - indexed_count, 0),
                unexpected=max(indexed_count - expected_count, 0),
            )

        healthy = not (
            missing_ids
            or unexpected_ids
            or duplicate_expected_ids
            or metadata_errors
            or any(
                coverage.missing or coverage.unexpected
                for coverage in source_coverage.values()
            )
        )
        return IndexIntegrity(
            healthy=healthy,
            expected_count=len(expected_map),
            indexed_count=len(actual_ids),
            missing_ids=missing_ids,
            unexpected_ids=unexpected_ids,
            duplicate_expected_ids=duplicate_expected_ids,
            metadata_errors=tuple(sorted(metadata_errors)),
            source_coverage=source_coverage,
        )

    # ---- indexing -------------------------------------------------------------------------
    def _rebuild_index(self) -> None:
        """Construct authoritative in-memory index from disk KB (Markdown + YAML).
        Merges persistent ChromaDB records when available.
        """
        # Load all authoritative raw chunks from disk
        raw = self._kb_chunks()
        self._chunks = {c.id: c for c in raw}

        # If ChromaDB collection exists and has additional or updated records, merge them
        if self._col is not None:
            try:
                data = self._col.get(include=["documents", "metadatas"])
                if data and data.get("ids"):
                    for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                        if cid not in self._chunks:
                            self._chunks[cid] = Chunk(
                                id=cid,
                                text=doc,
                                source=meta.get("source", ""),
                                section=meta.get("section", ""),
                                domain=meta.get("domain", "general"),
                                subtopic=meta.get("subtopic", ""),
                                jurisdiction=meta.get("jurisdiction", "national"),
                                authority=meta.get("authority", "official_government"),
                            )
            except Exception:
                log.warning("Could not merge ChromaDB records into index; relying on disk chunks", exc_info=True)

        self._bm25_ids = list(self._chunks)
        if BM25Okapi is not None and self._bm25_ids:
            corpus = [tokenize(self._chunks[i].text) for i in self._bm25_ids]
            self._bm25 = BM25Okapi(corpus) if corpus else None
        else:
            self._bm25 = None

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if self._client is None or types is None:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            try:
                r = self._client.models.embed_content(
                    model=self._s.embed_model,
                    contents=texts[i : i + EMBED_BATCH],
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=self._s.embed_dims
                    ),
                )
            except Exception as exc:
                self._client = None
                if (
                    genai_errors is not None
                    and isinstance(exc, genai_errors.ClientError)
                    and exc.code == 429
                ):
                    log.warning(
                        "Gemini embedding quota exhausted; using BM25-only retrieval until restart"
                    )
                else:
                    log.warning(
                        "Gemini embeddings unavailable (%s); using BM25-only retrieval until restart",
                        exc,
                    )
                raise
            out.extend(e.values for e in r.embeddings)
        return out

    def _kb_chunks(self) -> list[Chunk]:
        """Load knowledge chunks solely from main backend knowledge directory (kb_dir)."""
        chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        def _add_chunks(new_chunks: list[Chunk]):
            for c in new_chunks:
                if c.id not in seen_ids:
                    seen_ids.add(c.id)
                    chunks.append(c)

        # Ingest both YAML files and Markdown files from backend/kb/
        if self._s.kb_dir.exists():
            # 1. Structured YAML files in backend/kb/yaml/
            nested_yamls = sorted(self._s.kb_dir.rglob("*.yaml")) + sorted(self._s.kb_dir.rglob("*.yml"))
            for yf in nested_yamls:
                try:
                    data = yaml.safe_load(yf.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        _add_chunks(parse_yaml_chunks(data, yf.name))
                except Exception:
                    log.warning("Failed parsing backend YAML file %s", yf.name, exc_info=True)

            # 2. Markdown files in backend/kb/ subdirectories
            sub_md = [f for f in sorted(self._s.kb_dir.rglob("*.md")) if f.parent != self._s.kb_dir]
            md_files = sub_md if sub_md else sorted(self._s.kb_dir.glob("*.md"))
            for mf in md_files:
                try:
                    _add_chunks(parse_chunks(mf.read_text(encoding="utf-8"), mf.name))
                except Exception:
                    log.warning("Failed parsing Markdown file %s", mf.name, exc_info=True)

        return chunks

    def _fingerprint(self) -> str:
        h = hashlib.sha256(f"{self._s.embed_model}:{self._s.embed_dims}:v3".encode())

        # Main Backend KB
        if self._s.kb_dir.exists():
            for f in sorted(self._s.kb_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".md", ".yaml", ".yml"):
                    h.update(f.name.encode() + f.read_bytes())

        return h.hexdigest()

    def ingest(self) -> int:
        chunks = self._kb_chunks()
        if not chunks:
            return 0
        if self._col is None:
            self._rebuild_index()
            return len(chunks)

        expected = {c.id: c for c in chunks}
        existing_ids = self._collection_ids()
        existing_records = self._col.get(
            ids=sorted(existing_ids),
            include=["documents", "metadatas"],
        ) if existing_ids else {"ids": [], "documents": [], "metadatas": []}
        existing_documents = dict(
            zip(existing_records.get("ids") or [], existing_records.get("documents") or [])
        )
        existing_metadata = dict(
            zip(existing_records.get("ids") or [], existing_records.get("metadatas") or [])
        )

        to_upsert: list[Chunk] = []
        for cid, chunk_value in expected.items():
            if (
                cid not in existing_ids
                or existing_documents.get(cid) != chunk_value.text
                or existing_metadata.get(cid) != self._chunk_metadata(chunk_value)
            ):
                to_upsert.append(chunk_value)

        if to_upsert:
            try:
                embeddings = self._embed(
                    [c.text for c in to_upsert], "RETRIEVAL_DOCUMENT"
                )
            except Exception as exc:
                raise RuntimeError("Embedding generation failed; index unchanged") from exc
            if len(embeddings) != len(to_upsert):
                raise RuntimeError(
                    "Embedding generation returned an incomplete batch; index unchanged"
                )
            try:
                self._col.upsert(
                    ids=[c.id for c in to_upsert],
                    documents=[c.text for c in to_upsert],
                    metadatas=[self._chunk_metadata(c) for c in to_upsert],
                    embeddings=embeddings,
                )
            except Exception as exc:
                raise RuntimeError("Chroma upsert failed; index integrity not updated") from exc
            log.info("upserted %d changed knowledge chunks", len(to_upsert))

        stale = existing_ids - set(expected)
        if stale:
            try:
                self._col.delete(ids=sorted(stale))
            except Exception as exc:
                raise RuntimeError("Chroma stale-record cleanup failed") from exc
            log.info("removed %d obsolete knowledge chunks", len(stale))

        integrity = self.verify_index_integrity(chunks)
        if not integrity.healthy:
            raise RuntimeError(
                "Post-ingestion integrity verification failed: "
                f"missing={len(integrity.missing_ids)}, "
                f"unexpected={len(integrity.unexpected_ids)}, "
                f"metadata={len(integrity.metadata_errors)}"
            )
        self._s.chroma_dir.mkdir(parents=True, exist_ok=True)
        (self._s.chroma_dir / "kb_v2.sha256").write_text(
            self._fingerprint(), encoding="utf-8"
        )
        self._rebuild_index()
        log.info("rebuilt knowledge index with %d chunks from %d files", len(self._chunks), len({c.source for c in self._chunks.values()}))
        return len(self._chunks)

    def ensure_fresh(self) -> bool:
        """Re-ingest only when kb/*.yaml, kb/*.md or the embedding config changed."""
        marker = self._s.chroma_dir / "kb_v2.sha256"
        if self._col is None:
            self._rebuild_index()
            return False
        integrity = self.verify_index_integrity()
        if integrity.healthy and marker.exists():
            try:
                if marker.read_text(encoding="utf-8").strip() == self._fingerprint():
                    return False
            except OSError:
                pass
        self.ingest()
        return True

    # ---- retrieval ------------------------------------------------------------------------
    def _vector_search(self, query: str, allowed_domains: set[str] | None = None) -> list[tuple[str, float]]:
        if self._col is None or self._client is None:
            return []

        where_filter = None
        if allowed_domains:
            domain_list = sorted(allowed_domains)
            if len(domain_list) == 1:
                where_filter = {"domain": domain_list[0]}
            else:
                where_filter = {"domain": {"$in": domain_list}}

        try:
            embeddings = self._embed([query], "RETRIEVAL_QUERY")
            if not embeddings:
                return []
            r = self._col.query(
                query_embeddings=embeddings,
                where=where_filter,
                n_results=min(VEC_K, self.count()),
            )
            if r and r.get("ids") and r["ids"][0]:
                return list(zip(r["ids"][0], r["distances"][0]))
        except Exception:
            log.warning("filtered vector search failed, falling back to unfiltered", exc_info=True)

        try:
            r = self._col.query(
                query_embeddings=self._embed([query], "RETRIEVAL_QUERY"),
                n_results=min(VEC_K, self.count()),
            )
            return list(zip(r["ids"][0], r["distances"][0])) if r and r.get("ids") and r["ids"][0] else []
        except Exception:
            return []

    def _bm25_search(self, query: str, allowed_domains: set[str] | None = None) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda p: p[1], reverse=True)
        out = []
        for cid, s in ranked:
            if s <= 0:
                continue
            if allowed_domains and self._chunks.get(cid) and self._chunks[cid].domain not in allowed_domains:
                continue
            out.append((cid, float(s)))
            if len(out) >= BM25_K:
                break
        return out

    def _deterministic_rerank(self, query: str, detected_domains: list[str], cands: list[Chunk], k: int) -> list[Chunk]:
        """Fast, production-safe, deterministic multi-signal ranker. Zero external API calls."""
        if len(cands) <= 1:
            return cands[:k]

        q_tokens = set(tokenize(query))
        scored: list[tuple[float, Chunk]] = []

        for rank_idx, c in enumerate(cands):
            # Base RRF score
            base_score = 1.0 / (RRF_K + rank_idx + 1)
            # Domain bonus: strong preference if chunk matches detected query domain
            domain_bonus = 0.35 if c.domain in detected_domains else 0.0
            # Section bonus: preference if chunk section heading overlaps query
            sec_tokens = set(tokenize(c.section))
            sec_overlap = len(q_tokens & sec_tokens)
            sec_bonus = 0.15 if sec_overlap > 0 else 0.0
            # Text coverage
            chunk_tokens = set(tokenize(c.text))
            token_coverage = len(q_tokens & chunk_tokens) / max(len(q_tokens), 1)

            final_score = base_score + domain_bonus + sec_bonus + (0.20 * token_coverage)
            scored.append((final_score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    def _rerank(self, query: str, detected_domains: list[str], cands: list[Chunk], k: int) -> list[Chunk]:
        """Reranker: uses deterministic ranker by default; uses LLM rerank only if explicitly configured."""
        if not self._s.rerank or self._client is None or types is None:
            return self._deterministic_rerank(query, detected_domains, cands, k)

        if len(cands) <= 1:
            return cands[:k]
        passages = "\n\n".join(f"[{i}] {c.text[:RERANK_CHARS]}" for i, c in enumerate(cands))
        try:
            r = self._client.models.generate_content(
                model=self._s.rerank_model,
                contents=RERANK_PROMPT.format(query=query, passages=passages, k=k),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[int],
                    temperature=0,
                ),
            )
            order = json.loads(r.text)
            valid = [i for i in dict.fromkeys(order) if isinstance(i, int) and 0 <= i < len(cands)]
            if valid:
                return [cands[i] for i in valid][:k]
        except Exception:
            log.warning("LLM rerank failed; falling back to deterministic ranker", exc_info=True)

        return self._deterministic_rerank(query, detected_domains, cands, k)

    def search(self, query: str, k: int = 4, domains: list[str] | None = None) -> list[dict]:
        """Perform domain-aware, metadata-filtered hybrid retrieval with scope guarding."""
        if not query or not query.strip():
            return []

        if self.count() == 0:
            return []

        detected_domains = domains or classify_domains(query)
        if detected_domains == ["conversation"]:
            return []

        expanded = expand_query(query)

        # Build allowed domains set for metadata filtering
        allowed_domains: set[str] = set(detected_domains) | CROSS_CUTTING_DOMAINS
        # Strictly isolate financial fraud and social media crimes:
        if "financial_fraud" not in detected_domains:
            # If not financial fraud, do NOT pull financial fraud evidence/banking rules
            allowed_domains.discard("financial_fraud")
        if not any(d in detected_domains for d in ("social_media", "deepfake", "harassment", "sextortion")):
            # If purely financial or telecom, do NOT pull deepfake/social media
            allowed_domains.difference_update({"social_media", "deepfake", "harassment", "sextortion"})

        vec = self._vector_search(expanded, allowed_domains)
        lex = self._bm25_search(expanded, allowed_domains)

        if not vec and not lex:
            return []

        best_dist = vec[0][1] if vec else 1.0
        best_lex = lex[0][1] if lex else 0.0

        # Scope guard: reject if neither semantic distance nor lexical match meets threshold
        # (If vector search was unavailable e.g. offline, rely on calibrated lexical match)
        if vec:
            if best_dist > OOS_DISTANCE and best_lex < BM25_MIN_SCORE:
                log.debug("Scope guard triggered for query=%r (dist=%.3f, bm25=%.1f)", query, best_dist, best_lex)
                return []
        else:
            # When offline/lexical-only: out-of-domain queries require strong signal
            min_lex = BM25_MIN_SCORE if not detected_domains else 6.0
            if best_lex < min_lex:
                log.debug("Lexical scope guard triggered for query=%r (bm25=%.1f, min=%.1f)", query, best_lex, min_lex)
                return []

        fused = rrf([[i for i, _ in vec], [i for i, _ in lex]])[:RERANK_K]
        cands = [self._chunks[i] for i in fused if i in self._chunks]
        cands = _filter_platform_chunks(query, cands)
        if not cands:
            return []

        picked = self._rerank(query, detected_domains, cands, k)

        log.debug(
            "RAG search: query=%r | domains=%s | candidates=%d | picked=%d",
            query, detected_domains, len(cands), len(picked)
        )

        return [
            {
                "text": c.text,
                "source": c.source,
                "section": c.section,
                "domain": c.domain,
                "jurisdiction": c.jurisdiction,
                "authority": c.authority,
            }
            for c in picked
        ]


@lru_cache
def get_kb() -> KnowledgeBase:
    return KnowledgeBase(get_settings())


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    kb = get_kb()
    if kb.ensure_fresh():
        print(f"re-ingested fresh chunks; count={kb.count()}")
    else:
        print(f"KB is already fresh; count={kb.count()}")
    if len(sys.argv) > 1:
        for hit in kb.search(" ".join(sys.argv[1:])):
            print(f"\n[{hit['source']} > {hit['section']} ({hit['domain']})]\n{hit['text'][:300]}")
