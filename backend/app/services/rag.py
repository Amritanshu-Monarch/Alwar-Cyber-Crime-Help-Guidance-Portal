"""Hybrid RAG over kb/*.md.

Pipeline: heading-aware chunks (title + section prefixed) -> vector search (Gemini embeddings in Chroma)
+ lexical BM25 -> Reciprocal Rank Fusion -> optional LLM rerank -> scope guard (returns [] when the
question is unrelated, so the agent refuses instead of improvising)."""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

EMBED_BATCH = 50
CHUNK_CHARS = 900
VEC_K = 8
BM25_K = 8
RERANK_K = 6
RERANK_CHARS = 600
RRF_K = 60
# Scope guard: reject only when BOTH signals are weak. Calibrated with eval/rag_eval.py:
# in-scope queries had best distance <= 0.344, off-topic ones >= 0.430.
OOS_DISTANCE = 0.39  # best cosine distance above this = semantically far from every chunk
BM25_MIN_SCORE = 12.0  # ...unless there is very strong keyword overlap (generic words like "write" score ~6-8)
STOPWORDS = frozenset(
    "a an the and or of to in on for is are was were be been i my me you your it this that with at by from "
    "as do does did how what when where which who can should would will am not no".split()
)

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


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


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


def parse_chunks(markdown: str, source: str, size: int = CHUNK_CHARS) -> list[Chunk]:
    """Split on '## ' sections; every chunk is prefixed 'Title > Section' so it is self-describing."""
    title, heading, body = "", "", []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        if heading and "".join(body).strip():
            sections.append((heading, "\n".join(body).strip()))

    # files without '## ' sections use '# ' headings as the sections themselves
    flat = not re.search(r"^## ", markdown, re.MULTILINE)
    for line in markdown.splitlines():
        if line.startswith("## ") or (flat and line.startswith("# ")):
            flush()
            heading, body = line.lstrip("# ").strip(), []
        elif line.startswith("# "):
            title = line[2:].strip()
        else:
            body.append(line)
    flush()
    title = title or source.removesuffix(".md").replace("-", " ")

    stem = source.removesuffix(".md")
    chunks = []
    for s_idx, (sec, text) in enumerate(sections):
        for p_idx, piece in enumerate(chunk(text, size)):
            chunks.append(
                Chunk(f"{stem}-{s_idx}-{p_idx}", f"{title} > {sec}\n{piece}", source, sec)
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
        self._col = db.get_or_create_collection("alwar_kb", metadata={"hnsw:space": "cosine"})
        self._chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._rebuild_index()

    def count(self) -> int:
        return self._col.count()

    # ---- indexing -------------------------------------------------------------------------
    def _rebuild_index(self) -> None:
        data = self._col.get(include=["documents", "metadatas"])
        self._chunks = {
            cid: Chunk(cid, doc, meta["source"], meta.get("section", ""))
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
        for f in sorted(self._s.kb_dir.glob("*.md")):
            chunks += parse_chunks(f.read_text(encoding="utf-8"), f.name)
        return chunks

    def _fingerprint(self) -> str:
        h = hashlib.sha256(f"{self._s.embed_model}:{self._s.embed_dims}".encode())
        for f in sorted(self._s.kb_dir.glob("*.md")):
            h.update(f.name.encode() + f.read_bytes())
        return h.hexdigest()

    def ingest(self) -> int:
        chunks = self._kb_chunks()
        if not chunks:
            return 0
        stale = set(self._col.get()["ids"]) - {c.id for c in chunks}
        if stale:
            self._col.delete(ids=list(stale))
        self._col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "section": c.section} for c in chunks],
            embeddings=self._embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT"),
        )
        self._rebuild_index()
        (self._s.chroma_dir / "kb.sha256").write_text(self._fingerprint())
        log.info("ingested %d chunks from %d files", len(chunks), len({c.source for c in chunks}))
        return len(chunks)

    def ensure_fresh(self) -> bool:
        """Re-ingest only when kb/*.md or the embedding config changed. Returns True if it ingested."""
        marker = self._s.chroma_dir / "kb.sha256"
        if self.count() and marker.exists() and marker.read_text() == self._fingerprint():
            return False
        self.ingest()
        return True

    # ---- retrieval ------------------------------------------------------------------------
    def _vector_search(self, query: str) -> list[tuple[str, float]]:
        r = self._col.query(
            query_embeddings=self._embed([query], "RETRIEVAL_QUERY"),
            n_results=min(VEC_K, self.count()),
        )
        return list(zip(r["ids"][0], r["distances"][0]))

    def _bm25_search(self, query: str) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda p: p[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:BM25_K] if s > 0]

    def _rerank(self, query: str, cands: list[Chunk], k: int) -> list[Chunk]:
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
        except Exception:
            log.exception("rerank failed; using fused order")
            return cands[:k]
        valid = [i for i in dict.fromkeys(order) if isinstance(i, int) and 0 <= i < len(cands)]
        return [cands[i] for i in valid][:k]

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.count() == 0:
            return []
        vec = self._vector_search(query)
        lex = self._bm25_search(query)
        if not vec:
            return []
        best_lex = lex[0][1] if lex else 0.0
        if vec[0][1] > OOS_DISTANCE and best_lex < BM25_MIN_SCORE:
            return []  # unrelated to anything we know: let the agent refuse
        fused = rrf([[i for i, _ in vec], [i for i, _ in lex]])[:RERANK_K]
        cands = [self._chunks[i] for i in fused if i in self._chunks]
        picked = self._rerank(query, cands, k) if self._s.rerank else cands[:k]
        return [{"text": c.text, "source": c.source, "section": c.section} for c in picked]


@lru_cache
def get_kb() -> KnowledgeBase:
    return KnowledgeBase(get_settings())


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    kb = get_kb()
    print(f"ingested {kb.ingest()} chunks")
    if len(sys.argv) > 1:
        for hit in kb.search(" ".join(sys.argv[1:])):
            print(f"\n[{hit['source']} > {hit['section']}]\n{hit['text'][:300]}")
