from src.audit import audit_processed_slice


def test_audit_reports_long_text_missing_evidence_and_unanswerable_counts():
    papers = [
        {
            "paper_id": "p1",
            "paragraphs": [
                {"paragraph_id": "p1::p0000", "text": "short evidence paragraph"},
                {"paragraph_id": "p1::p0001", "text": " ".join(["long"] * 12)},
            ],
        }
    ]
    qas = [
        {
            "question_id": "p1::q1",
            "paper_id": "p1",
            "evidence": ["short evidence paragraph"],
            "evidence_ids": ["p1::p0000"],
            "unanswerable": False,
        },
        {
            "question_id": "p1::q2",
            "paper_id": "p1",
            "evidence": ["not found"],
            "evidence_ids": [],
            "unanswerable": False,
        },
        {
            "question_id": "p1::q3",
            "paper_id": "p1",
            "evidence": [],
            "evidence_ids": [],
            "unanswerable": True,
        },
    ]

    audit = audit_processed_slice(papers, qas, long_paragraph_words=10)

    assert audit["papers"] == 1
    assert audit["qas"] == 3
    assert audit["paragraphs"] == 2
    assert audit["long_paragraphs"]["count"] == 1
    assert audit["evidence"]["missing_or_incomplete_count"] == 1
    assert audit["unanswerable"]["count"] == 1
    assert audit["unanswerable"]["share"] == 1 / 3
