from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def audit_processed_slice(
    papers: list[dict[str, Any]],
    qas: list[dict[str, Any]],
    long_document_words: int = 4000,
    long_paragraph_words: int = 250,
) -> dict[str, Any]:
    paragraph_lengths: list[int] = []
    document_lengths: list[int] = []
    long_paragraph_ids: list[str] = []
    long_document_ids: list[str] = []

    for paper in papers:
        doc_words = 0
        for paragraph in paper.get("paragraphs", []):
            words = _word_count(paragraph.get("text", ""))
            paragraph_lengths.append(words)
            doc_words += words
            if words >= long_paragraph_words:
                long_paragraph_ids.append(str(paragraph.get("paragraph_id")))
        document_lengths.append(doc_words)
        if doc_words >= long_document_words:
            long_document_ids.append(str(paper.get("paper_id")))

    exact_evidence_matches = 0
    partial_evidence_matches = 0
    missing_evidence_matches = 0
    missing_or_incomplete: list[dict[str, Any]] = []
    for qa in qas:
        matches = qa.get("evidence_matches") or []
        if matches:
            exact_evidence_matches += sum(1 for match in matches if match.get("match_type") == "exact")
            partial_evidence_matches += sum(1 for match in matches if match.get("match_type") == "partial")
            missing_evidence_matches += sum(1 for match in matches if match.get("match_type") == "missing")
            if not qa.get("unanswerable") and any(match.get("match_type") == "missing" for match in matches):
                missing_or_incomplete.append(qa)
        elif not qa.get("unanswerable") and bool(qa.get("evidence")) and not bool(qa.get("evidence_ids")):
            missing_evidence_matches += len(qa.get("evidence", []))
            missing_or_incomplete.append(qa)
    unanswerable = [qa for qa in qas if qa.get("unanswerable")]

    return {
        "papers": len(papers),
        "qas": len(qas),
        "paragraphs": len(paragraph_lengths),
        "document_lengths": _length_stats(document_lengths),
        "paragraph_lengths": _length_stats(paragraph_lengths),
        "long_documents": {
            "threshold_words": long_document_words,
            "count": len(long_document_ids),
            "paper_ids": long_document_ids,
        },
        "long_paragraphs": {
            "threshold_words": long_paragraph_words,
            "count": len(long_paragraph_ids),
            "paragraph_ids": long_paragraph_ids,
        },
        "evidence": {
            "exact_match_count": exact_evidence_matches,
            "partial_match_count": partial_evidence_matches,
            "missing_match_count": missing_evidence_matches,
            "missing_or_incomplete_count": len(missing_or_incomplete),
            "missing_or_incomplete_question_ids": [
                str(qa.get("question_id")) for qa in missing_or_incomplete
            ],
        },
        "unanswerable": {
            "count": len(unanswerable),
            "share": len(unanswerable) / len(qas) if qas else 0.0,
            "question_ids": [str(qa.get("question_id")) for qa in unanswerable],
        },
    }


def write_audit(audit: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _word_count(text: Any) -> int:
    return len(str(text).split())


def _length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": mean(values)}
