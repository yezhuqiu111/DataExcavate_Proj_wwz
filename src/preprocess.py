from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def normalize_qasper_records(
    raw_records: Iterable[dict[str, Any]],
    max_papers: int | None = None,
    max_qas: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize QASPER-like records into stable paper and QA JSON objects."""
    papers: list[dict[str, Any]] = []
    qas: list[dict[str, Any]] = []

    for raw_index, raw in enumerate(raw_records):
        if max_papers is not None and len(papers) >= max_papers:
            break
        if max_qas is not None and len(qas) >= max_qas:
            break

        paper_id = str(raw.get("id") or raw.get("paper_id") or f"paper-{raw_index:04d}")
        paragraphs = _normalize_paragraphs(paper_id, raw)
        paper = {
            "paper_id": paper_id,
            "title": str(raw.get("title") or ""),
            "abstract": _abstract_text(raw.get("abstract")),
            "paragraphs": paragraphs,
        }
        papers.append(paper)

        paragraph_lookup = [
            {
                "paragraph_id": paragraph["paragraph_id"],
                "text": paragraph["text"].strip(),
                "normalized_text": _normalize_text_key(paragraph["text"]),
            }
            for paragraph in paragraphs
            if paragraph["text"].strip()
        ]
        for qa_index, raw_qa in enumerate(_iter_qas(raw.get("qas", []))):
            if max_qas is not None and len(qas) >= max_qas:
                break
            qas.append(_normalize_qa(paper_id, qa_index, raw_qa, paragraph_lookup))

    return papers, qas


def write_processed_slice(
    papers: list[dict[str, Any]],
    qas: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paper_path = output_path / "papers.jsonl"
    qa_path = output_path / "qas.jsonl"
    _write_jsonl(paper_path, papers)
    _write_jsonl(qa_path, qas)
    return paper_path, qa_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_paragraphs(paper_id: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    for section_name, text in _iter_section_paragraphs(raw.get("full_text", [])):
        paragraph_id = f"{paper_id}::p{len(paragraphs):04d}"
        paragraphs.append(
            {
                "paragraph_id": paragraph_id,
                "paper_id": paper_id,
                "section": section_name,
                "text": text,
            }
        )
    return paragraphs


def _iter_section_paragraphs(full_text: Any) -> Iterable[tuple[str, str]]:
    if isinstance(full_text, dict):
        section_names = full_text.get("section_name") or full_text.get("section_names") or []
        paragraph_groups = full_text.get("paragraphs") or []
        for index, paragraph_group in enumerate(paragraph_groups):
            section_name = _section_name_at(section_names, index)
            for paragraph in _flatten_paragraph_group(paragraph_group):
                yield section_name, paragraph
        return

    if isinstance(full_text, list):
        for index, section in enumerate(full_text):
            if isinstance(section, dict):
                section_name = str(section.get("section_name") or section.get("name") or f"section-{index}")
                for paragraph in _flatten_paragraph_group(section.get("paragraphs", [])):
                    yield section_name, paragraph
            elif isinstance(section, str) and section.strip():
                yield f"section-{index}", section.strip()


def _section_name_at(section_names: Any, index: int) -> str:
    if isinstance(section_names, list) and index < len(section_names):
        return str(section_names[index])
    return f"section-{index}"


def _flatten_paragraph_group(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
        return
    if isinstance(value, list):
        for item in value:
            yield from _flatten_paragraph_group(item)


def _iter_qas(raw_qas: Any) -> Iterable[dict[str, Any]]:
    if isinstance(raw_qas, list):
        for item in raw_qas:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(raw_qas, dict):
        questions = raw_qas.get("question", [])
        question_ids = raw_qas.get("question_id", [])
        answers = raw_qas.get("answers", [])
        for index, question in enumerate(questions):
            yield {
                "question_id": _list_get(question_ids, index, f"q{index:04d}"),
                "question": question,
                "answers": _list_get(answers, index, []),
            }


def _normalize_qa(
    paper_id: str,
    qa_index: int,
    raw_qa: dict[str, Any],
    paragraph_lookup: list[dict[str, str]],
) -> dict[str, Any]:
    raw_question_id = str(raw_qa.get("question_id") or f"q{qa_index:04d}")
    answers, evidence_texts, unanswerable = _extract_answer_fields(raw_qa.get("answers", []))
    evidence_matches = [_match_evidence_text(text, paragraph_lookup) for text in evidence_texts]
    evidence_ids = _dedupe(
        [
            match["paragraph_id"]
            for match in evidence_matches
            if match["paragraph_id"] and match["match_type"] != "missing"
        ]
    )
    return {
        "question_id": f"{paper_id}::{raw_question_id}",
        "paper_id": paper_id,
        "question": str(raw_qa.get("question") or ""),
        "answers": answers,
        "evidence": evidence_texts,
        "evidence_ids": evidence_ids,
        "evidence_matches": evidence_matches,
        "unanswerable": unanswerable,
    }


def _extract_answer_fields(raw_answers: Any) -> tuple[list[str], list[str], bool]:
    answers: list[str] = []
    evidence_texts: list[str] = []
    unanswerable = False

    for item in _normalize_answer_items(raw_answers):
        answer = item.get("answer", item) if isinstance(item, dict) else item
        if not isinstance(answer, dict):
            continue
        unanswerable = unanswerable or bool(answer.get("unanswerable", False))
        for span in answer.get("extractive_spans") or []:
            if str(span).strip():
                answers.append(str(span).strip())
        free_form = str(answer.get("free_form_answer") or "").strip()
        if free_form:
            answers.append(free_form)
        yes_no = answer.get("yes_no")
        if yes_no is True:
            answers.append("yes")
        elif yes_no is False:
            answers.append("no")
        for evidence in answer.get("evidence") or []:
            if str(evidence).strip():
                evidence_texts.append(str(evidence).strip())

    return _dedupe(answers), _dedupe(evidence_texts), unanswerable


def _normalize_answer_items(raw_answers: Any) -> list[dict[str, Any]]:
    """Convert QASPER answer payloads into a list of annotator-style dicts."""
    if raw_answers is None:
        return []
    if isinstance(raw_answers, list):
        if not raw_answers:
            return []
        first = raw_answers[0]
        if isinstance(first, dict) and _is_columnar_answer_bundle(first):
            items: list[dict[str, Any]] = []
            for bundle in raw_answers:
                if isinstance(bundle, dict):
                    items.extend(_columnar_answer_bundle_to_items(bundle))
            return items
        if isinstance(first, dict) and _is_answer_payload(first):
            return [{"answer": item} for item in raw_answers if isinstance(item, dict)]
        return [item for item in raw_answers if isinstance(item, dict)]
    if isinstance(raw_answers, dict):
        if _is_columnar_answer_bundle(raw_answers):
            return _columnar_answer_bundle_to_items(raw_answers)
        if _is_answer_payload(raw_answers):
            return [{"answer": raw_answers}]
    return []


def _is_columnar_answer_bundle(value: dict[str, Any]) -> bool:
    return isinstance(value.get("answer"), list)


def _is_answer_payload(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("extractive_spans", "free_form_answer", "evidence", "unanswerable", "yes_no"))


def _columnar_answer_bundle_to_items(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    answer_values = bundle.get("answer") or []
    if not isinstance(answer_values, list):
        return []
    annotation_ids = bundle.get("annotation_id") or []
    worker_ids = bundle.get("worker_id") or []
    items: list[dict[str, Any]] = []
    for index, answer in enumerate(answer_values):
        if not isinstance(answer, dict):
            continue
        item: dict[str, Any] = {"answer": answer}
        if isinstance(annotation_ids, list) and index < len(annotation_ids):
            item["annotation_id"] = annotation_ids[index]
        if isinstance(worker_ids, list) and index < len(worker_ids):
            item["worker_id"] = worker_ids[index]
        items.append(item)
    return items


def _abstract_text(abstract: Any) -> str:
    if isinstance(abstract, str):
        return abstract
    if isinstance(abstract, list):
        return " ".join(str(part) for part in abstract if str(part).strip())
    return ""


def _list_get(value: Any, index: int, default: Any) -> Any:
    if isinstance(value, list) and index < len(value):
        return value[index]
    return default


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _match_evidence_text(evidence_text: str, paragraph_lookup: list[dict[str, str]]) -> dict[str, str]:
    text = evidence_text.strip()
    normalized = _normalize_text_key(text)
    if not normalized:
        return {"evidence": text, "paragraph_id": "", "match_type": "missing"}

    for paragraph in paragraph_lookup:
        if normalized == paragraph["normalized_text"]:
            return {"evidence": text, "paragraph_id": paragraph["paragraph_id"], "match_type": "exact"}

    for paragraph in paragraph_lookup:
        paragraph_text = paragraph["normalized_text"]
        if normalized in paragraph_text or paragraph_text in normalized:
            return {"evidence": text, "paragraph_id": paragraph["paragraph_id"], "match_type": "partial"}

    return {"evidence": text, "paragraph_id": "", "match_type": "missing"}


def _normalize_text_key(text: Any) -> str:
    return " ".join(str(text).lower().split())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
