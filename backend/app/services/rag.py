"""Domain-aware Hybrid RAG over kb/**/*.md.

Pipeline:
1. Frontmatter & heading-aware atomic chunks (with domain, subtopic, jurisdiction metadata).
2. Domain classification & Hindi/Hinglish query expansion.
3. Metadata-filtered Vector search (Gemini embeddings in ChromaDB) + Unicode-aware BM25.
4. Reciprocal Rank Fusion (RRF).
5. High-precision deterministic multi-signal reranking (+ optional LLM rerank).
6. Calibrated scope guard preventing out-of-domain hallucinations.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.services.query_expander import classify_domains, expand_query, tokenize

log = logging.getLogger(__name__)

EMBED_BATCH = 50
CHUNK_CHARS = 900
VEC_K = 8
BM25_K = 8
RERANK_K = 6
RERANK_CHARS = 600
RRF_K = 60

# Calibrated scope guard: reject when query has weak semantic similarity AND low lexical overlap
OOS_DISTANCE = 0.39
BM25_MIN_SCORE = 8.0

# General cross-domain fallback categories always eligible for supporting context
CROSS_CUTTING_DOMAINS = frozenset({"general", "reporting", "evidence", "local_police", "emergency"})

RERANK_PROMPT = """You judge relevance for a cyber crime help knowledge base used by Indian citizens.
Question: {query}

Passages:
{passages}

Return a JSON array of the passage numbers that directly help answer the question, most helpful first
(at most {k}). Return [] if none of them are relevant to the question."""


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


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: merge several best-first id lists into one ordering."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


class KnowledgeBase:
    def __init__(self, settings: Settings, client: genai.Client | None = None):
        self._s = settings
        self._client = client or genai.Client(api_key=settings.google_api_key or None)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        db = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._col = db.get_or_create_collection("alwar_kb_v2", metadata={"hnsw:space": "cosine"})
        self._chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._rebuild_index()

    def count(self) -> int:
        return self._col.count()

    # ---- indexing -------------------------------------------------------------------------
    def _rebuild_index(self) -> None:
        data = self._col.get(include=["documents", "metadatas"])
        if not data or not data.get("ids"):
            self._chunks = {}
            self._bm25 = None
            self._bm25_ids = []
            return

        self._chunks = {
            cid: Chunk(
                id=cid,
                text=doc,
                source=meta.get("source", ""),
                section=meta.get("section", ""),
                domain=meta.get("domain", "general"),
                subtopic=meta.get("subtopic", ""),
                jurisdiction=meta.get("jurisdiction", "national"),
                authority=meta.get("authority", "official_government"),
            )
            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        }
        self._bm25_ids = list(self._chunks)
        corpus = [tokenize(self._chunks[i].text) for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            r = self._client.models.embed_content(
                model=self._s.embed_model,
                contents=texts[i : i + EMBED_BATCH],
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self._s.embed_dims
                ),
            )
            out.extend(e.values for e in r.embeddings)
        return out

    def _kb_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        # Prefer structured atomic files in subdirectories
        sub_files = [f for f in sorted(self._s.kb_dir.rglob("*.md")) if f.parent != self._s.kb_dir]
        files = sub_files if sub_files else sorted(self._s.kb_dir.glob("*.md"))
        for f in files:
            chunks += parse_chunks(f.read_text(encoding="utf-8"), f.name)
        return chunks

    def _fingerprint(self) -> str:
        h = hashlib.sha256(f"{self._s.embed_model}:{self._s.embed_dims}:v2".encode())
        sub_files = [f for f in sorted(self._s.kb_dir.rglob("*.md")) if f.parent != self._s.kb_dir]
        files = sub_files if sub_files else sorted(self._s.kb_dir.glob("*.md"))
        for f in files:
            h.update(f.name.encode() + f.read_bytes())
        return h.hexdigest()

    def ingest(self) -> int:
        chunks = self._kb_chunks()
        if not chunks:
            return 0
        existing_ids = set(self._col.get()["ids"])
        new_ids = {c.id for c in chunks}
        stale = existing_ids - new_ids
        if stale:
            self._col.delete(ids=list(stale))
        self._col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "section": c.section,
                    "domain": c.domain,
                    "subtopic": c.subtopic,
                    "jurisdiction": c.jurisdiction,
                    "authority": c.authority,
                }
                for c in chunks
            ],
            embeddings=self._embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT"),
        )
        self._rebuild_index()
        (self._s.chroma_dir / "kb_v2.sha256").write_text(self._fingerprint())
        log.info("ingested %d chunks from %d files", len(chunks), len({c.source for c in chunks}))
        return len(chunks)

    def ensure_fresh(self) -> bool:
        """Re-ingest only when kb/*.md or the embedding config changed. Returns True if it ingested."""
        marker = self._s.chroma_dir / "kb_v2.sha256"
        if self.count() and marker.exists() and marker.read_text() == self._fingerprint():
            return False
        self.ingest()
        return True

    # ---- retrieval ------------------------------------------------------------------------
    def _vector_search(self, query: str, allowed_domains: set[str] | None = None) -> list[tuple[str, float]]:
        where_filter = None
        if allowed_domains:
            domain_list = sorted(allowed_domains)
            if len(domain_list) == 1:
                where_filter = {"domain": domain_list[0]}
            else:
                where_filter = {"domain": {"$in": domain_list}}

        try:
            r = self._col.query(
                query_embeddings=self._embed([query], "RETRIEVAL_QUERY"),
                where=where_filter,
                n_results=min(VEC_K, self.count()),
            )
            if r and r.get("ids") and r["ids"][0]:
                return list(zip(r["ids"][0], r["distances"][0]))
        except Exception:
            log.warning("filtered vector search failed, falling back to unfiltered", exc_info=True)

        r = self._col.query(
            query_embeddings=self._embed([query], "RETRIEVAL_QUERY"),
            n_results=min(VEC_K, self.count()),
        )
        return list(zip(r["ids"][0], r["distances"][0])) if r and r.get("ids") and r["ids"][0] else []

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
        if not self._s.rerank:
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
        if best_dist > OOS_DISTANCE and best_lex < BM25_MIN_SCORE:
            log.debug("Scope guard triggered for query=%r (dist=%.3f, bm25=%.1f)", query, best_dist, best_lex)
            return []

        fused = rrf([[i for i, _ in vec], [i for i, _ in lex]])[:RERANK_K]
        cands = [self._chunks[i] for i in fused if i in self._chunks]

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
