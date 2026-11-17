"""Calibration and evaluation tests for Intent Classification & Scope Guard.
Evaluates precision and recall across 150 labelled queries in backend/tests/data/scope_eval.jsonl.
"""

import json
from pathlib import Path
from collections import defaultdict

import pytest

from app.services.intent_classifier import classify_intent, IntentOutcome

DATA_FILE = Path(__file__).resolve().parent / "data" / "scope_eval.jsonl"


def load_dataset() -> list[dict]:
    records = []
    assert DATA_FILE.exists(), f"Evaluation dataset not found at {DATA_FILE}"
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_scope_eval_dataset_integrity():
    dataset = load_dataset()
    assert len(dataset) >= 150, f"Expected at least 150 evaluation queries, found {len(dataset)}"

    languages = set(r["lang"] for r in dataset)
    assert {"en", "hi", "hinglish"}.issubset(languages)

    labels = set(r["label"] for r in dataset)
    assert {"in_scope", "out_of_scope", "needs_clarification"}.issubset(labels)


def test_classify_intent_accuracy_and_report():
    dataset = load_dataset()
    
    # Track statistics per language
    # {lang: {label: {"tp": 0, "fp": 0, "fn": 0, "total": 0}}}
    stats = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0}))
    total_correct = 0

    for item in dataset:
        query = item["query"]
        expected = item["label"]
        lang = item["lang"]

        pred = classify_intent(query, lang=lang)
        actual = pred.status

        stats[lang][expected]["total"] += 1
        if actual == expected:
            total_correct += 1
            stats[lang][expected]["tp"] += 1
        else:
            stats[lang][expected]["fn"] += 1
            stats[lang][actual]["fp"] += 1

    overall_accuracy = (total_correct / len(dataset)) * 100
    print(f"\n=======================================================")
    print(f"📊 SCOPE & INTENT CLASSIFICATION CALIBRATION REPORT")
    print(f"Total Evaluated Queries: {len(dataset)}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{len(dataset)})")
    print(f"=======================================================")

    for lang in ("en", "hi", "hinglish"):
        print(f"\nLanguage: {lang.upper()}")
        for label in ("in_scope", "out_of_scope", "needs_clarification"):
            s = stats[lang][label]
            tp = s["tp"]
            fp = s["fp"]
            fn = s["fn"]
            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
            print(f"  • {label:20s} | Precision: {precision:6.2f}% | Recall: {recall:6.2f}% | (Count: {s['total']})")

    # Enforce quality gates
    assert overall_accuracy >= 92.0, f"Overall classification accuracy {overall_accuracy:.2f}% below 92% target"
    for lang in ("en", "hi", "hinglish"):
        # Critical safety rule: In-scope queries must have high recall (no false rejection of real cybercrime victims)
        in_scope_recall = (stats[lang]["in_scope"]["tp"] / (stats[lang]["in_scope"]["tp"] + stats[lang]["in_scope"]["fn"])) * 100
        assert in_scope_recall >= 90.0, f"In-scope recall for {lang} ({in_scope_recall:.2f}%) below 90%"


def test_borderline_queries_never_reject_silently():
    """Verify borderline queries receive targeted clarifying questions instead of silent abstention."""
    ambiguous_examples = [
        ("Someone is calling me repeatedly", "en"),
        ("मेरे फोन पर एक अजीब सा लिंक आया था मैंने खोल दिया", "hi"),
        ("Ek link pe touch ho gaya tha screen pe", "hinglish"),
    ]
    for query, lang in ambiguous_examples:
        res = classify_intent(query, lang=lang)
        assert res.status == "needs_clarification"
        assert len(res.clarification_question) > 20
        assert "1930" in res.clarification_question or "ऑनलाइन" in res.clarification_question or "online" in res.clarification_question
