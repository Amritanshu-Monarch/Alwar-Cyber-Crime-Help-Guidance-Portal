from app.services.rag import chunk, parse_chunks, rrf, tokenize

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


def test_chunk_packs_paragraphs_and_respects_size():
    text = "\n\n".join(f"para {i} " + "x" * 200 for i in range(6))
    out = chunk(text, size=500)
    assert len(out) > 1
    assert all(len(c) <= 700 for c in out)
    assert "para 0" in out[0] and "para 5" in out[-1]


def test_chunk_empty():
    assert chunk("") == []
