from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.answering import INSUFFICIENT_EVIDENCE
from src.retrieval import TOKEN_RE


def evaluate_predictions(
    qas: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    prediction_by_id = {prediction.get("question_id"): prediction for prediction in predictions}
    recall_hits = 0
    recall_total = 0
    exact_match_values: list[float] = []
    f1_values: list[float] = []
    evidence_precision_values: list[float] = []
    evidence_recall_values: list[float] = []
    evidence_f1_values: list[float] = []
    refusal_hits = 0
    refusal_total = 0
    unsupported_claims = 0
    supported_answer_total = 0
    latencies: list[float] = []

    for qa in qas:
        prediction = prediction_by_id.get(qa.get("question_id"), {})
        retrieved_ids = prediction.get("retrieved_evidence_ids", [])[:top_k]
        gold_ids = qa.get("evidence_ids", [])
        if gold_ids:
            recall_total += 1
            if set(gold_ids) & set(retrieved_ids):
                recall_hits += 1
            precision, recall, f1 = _evidence_prf(retrieved_ids, gold_ids)
            evidence_precision_values.append(precision)
            evidence_recall_values.append(recall)
            evidence_f1_values.append(f1)

        if qa.get("answers"):
            predicted_answer = str(prediction.get("predicted_answer", ""))
            f1_values.append(_best_answer_f1(predicted_answer, qa.get("answers", [])))
            exact_match_values.append(_best_exact_match(predicted_answer, qa.get("answers", [])))

        if qa.get("unanswerable"):
            refusal_total += 1
            if prediction.get("refused") or prediction.get("predicted_answer") == INSUFFICIENT_EVIDENCE:
                refusal_hits += 1

        predicted_refusal = prediction.get("refused") or prediction.get("predicted_answer") == INSUFFICIENT_EVIDENCE
        if not predicted_refusal:
            supported_answer_total += 1
            if not _answer_supported_by_evidence(
                str(prediction.get("predicted_answer", "")),
                prediction.get("retrieved_evidence", []),
            ):
                unsupported_claims += 1

        if prediction.get("latency_ms") is not None:
            latencies.append(float(prediction["latency_ms"]))

    evidence_recall = recall_hits / recall_total if recall_total else 0.0
    metrics = {
        "questions": len(qas),
        "top_k": top_k,
        "evidence_recall_at_k": evidence_recall,
        f"evidence_recall_at_{top_k}": evidence_recall,
        "evidence_precision_at_k": sum(evidence_precision_values) / len(evidence_precision_values)
        if evidence_precision_values
        else 0.0,
        "evidence_id_recall_at_k": sum(evidence_recall_values) / len(evidence_recall_values)
        if evidence_recall_values
        else 0.0,
        "evidence_f1_at_k": sum(evidence_f1_values) / len(evidence_f1_values) if evidence_f1_values else 0.0,
        "answer_exact_match": sum(exact_match_values) / len(exact_match_values) if exact_match_values else 0.0,
        "answer_token_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "refusal_accuracy": refusal_hits / refusal_total if refusal_total else 0.0,
        "unsupported_claim_rate": unsupported_claims / supported_answer_total if supported_answer_total else 0.0,
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }
    if top_k != 5:
        metrics["evidence_recall_at_5"] = None
    return metrics


def select_failure_cases(
    qas: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    limit: int = 2,
) -> list[dict[str, Any]]:
    prediction_by_id = {prediction.get("question_id"): prediction for prediction in predictions}
    failures: list[dict[str, Any]] = []

    for qa in qas:
        prediction = prediction_by_id.get(qa.get("question_id"), {})
        reason = _failure_reason(qa, prediction)
        if not reason:
            continue
        failures.append(
            {
                "question_id": qa.get("question_id"),
                "question": qa.get("question", prediction.get("question", "")),
                "prediction": prediction.get("predicted_answer", ""),
                "reference_answers": qa.get("answers", []),
                "gold_evidence_ids": qa.get("evidence_ids", []),
                "retrieved_evidence_ids": prediction.get("retrieved_evidence_ids", []),
                "retrieved_evidence_preview": [
                    {
                        "paragraph_id": item.get("paragraph_id"),
                        "score": item.get("score"),
                        "text": str(item.get("text", ""))[:300],
                    }
                    for item in prediction.get("retrieved_evidence", [])[:3]
                ],
                "graph_trace": prediction.get("graph_trace"),
                "failure_reason": reason,
                "improvement_direction": _improvement_direction(reason),
            }
        )
        if len(failures) >= limit:
            break

    return failures


def _failure_reason(qa: dict[str, Any], prediction: dict[str, Any]) -> str | None:
    predicted_refusal = prediction.get("refused") or prediction.get("predicted_answer") == INSUFFICIENT_EVIDENCE
    if qa.get("unanswerable"):
        return None if predicted_refusal else "refusal_miss"

    gold_ids = set(qa.get("evidence_ids", []))
    retrieved_ids = set(prediction.get("retrieved_evidence_ids", []))
    if gold_ids and not (gold_ids & retrieved_ids):
        return "retrieval_miss"
    if predicted_refusal:
        return "over_refusal"
    if qa.get("answers") and _best_answer_f1(str(prediction.get("predicted_answer", "")), qa.get("answers", [])) == 0.0:
        return "answer_mismatch"
    return None


def _improvement_direction(reason: str) -> str:
    directions = {
        "retrieval_miss": "Improve paragraph ranking or graph expansion so annotated evidence enters top-k.",
        "over_refusal": "Lower refusal threshold or add graph matches before refusing answerable questions.",
        "refusal_miss": "Raise refusal threshold for weak retrieval on unanswerable questions.",
        "answer_mismatch": "Improve sentence selection from retrieved evidence.",
    }
    return directions.get(reason, "Inspect retrieval and answer extraction for this question.")


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _best_answer_f1(predicted: str, answers: list[str]) -> float:
    return max((_token_f1(predicted, answer) for answer in answers), default=0.0)


def _best_exact_match(predicted: str, answers: list[str]) -> float:
    normalized = _normalize_answer(predicted)
    return 1.0 if any(normalized == _normalize_answer(answer) for answer in answers) else 0.0


def _token_f1(predicted: str, reference: str) -> float:
    predicted_tokens = [token.lower() for token in TOKEN_RE.findall(predicted)]
    reference_tokens = [token.lower() for token in TOKEN_RE.findall(reference)]
    if not predicted_tokens or not reference_tokens:
        return 0.0
    overlap = 0
    remaining = reference_tokens.copy()
    for token in predicted_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _evidence_prf(retrieved_ids: list[str], gold_ids: list[str]) -> tuple[float, float, float]:
    retrieved = set(retrieved_ids)
    gold = set(gold_ids)
    if not retrieved or not gold:
        return 0.0, 0.0, 0.0
    overlap = len(retrieved & gold)
    precision = overlap / len(retrieved)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _answer_supported_by_evidence(predicted: str, evidence: list[dict[str, Any]]) -> bool:
    answer_tokens = {
        token.lower()
        for token in TOKEN_RE.findall(predicted)
        if len(token) >= 4 and token.lower() not in {"this", "that", "with", "from", "they", "have"}
    }
    if not answer_tokens:
        return False
    evidence_tokens = {
        token.lower()
        for item in evidence
        for token in TOKEN_RE.findall(str(item.get("text", "")))
    }
    return bool(answer_tokens & evidence_tokens)


def _normalize_answer(text: str) -> str:
    return " ".join(token.lower() for token in TOKEN_RE.findall(text))
