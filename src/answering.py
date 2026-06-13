from __future__ import annotations

import re
from typing import Any

from src.retrieval import TOKEN_RE


INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def answer_from_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    min_score: float = 0.12,
    has_graph_match: bool = False,
    query_term_overlap: int = 0,
    enable_refusal: bool = True,
) -> dict[str, Any]:
    if not evidence:
        return {"answer": INSUFFICIENT_EVIDENCE, "evidence_ids": [], "refused": True}

    support_score = float(evidence[0].get("lexical_score", evidence[0].get("score", 0.0)))
    effective_min_score = min_score * 0.85 if has_graph_match and query_term_overlap >= 1 else min_score
    if enable_refusal and support_score < effective_min_score:
        return {"answer": INSUFFICIENT_EVIDENCE, "evidence_ids": [], "refused": True}
    if enable_refusal and support_score < min_score and query_term_overlap < 1:
        return {"answer": INSUFFICIENT_EVIDENCE, "evidence_ids": [], "refused": True}

    sentence = _best_sentence(question, str(evidence[0].get("text", "")))
    return {
        "answer": sentence,
        "evidence_ids": [evidence[0].get("paragraph_id")],
        "refused": False,
    }


def _best_sentence(question: str, text: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return text.strip()
    query_terms = {token.lower() for token in TOKEN_RE.findall(question)}
    return max(
        sentences,
        key=lambda sentence: (
            len(query_terms & {token.lower() for token in TOKEN_RE.findall(sentence)}),
            -len(sentence),
        ),
    )
