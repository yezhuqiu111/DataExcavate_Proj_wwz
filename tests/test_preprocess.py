import json

from src.preprocess import normalize_qasper_records, write_processed_slice


def test_tiny_qasper_fixture_writes_stable_processed_jsonl(tmp_path):
    raw_records = [
        {
            "id": "paper-a",
            "title": "Graph retrieval for papers",
            "abstract": "We study evidence retrieval.",
            "full_text": [
                {
                    "section_name": "Introduction",
                    "paragraphs": [
                        "Graph retrieval links terms in paper paragraphs.",
                        "Unrelated baseline text.",
                    ],
                }
            ],
            "qas": [
                {
                    "question_id": "q-a",
                    "question": "What links terms?",
                    "answers": [
                        {
                            "answer": {
                                "extractive_spans": ["Graph retrieval"],
                                "free_form_answer": "",
                                "evidence": [
                                    "Graph retrieval links terms in paper paragraphs."
                                ],
                                "unanswerable": False,
                            }
                        }
                    ],
                }
            ],
        }
    ]

    papers, qas = normalize_qasper_records(raw_records)
    write_processed_slice(papers, qas, tmp_path)

    paper_lines = (tmp_path / "papers.jsonl").read_text().splitlines()
    qa_lines = (tmp_path / "qas.jsonl").read_text().splitlines()

    assert len(paper_lines) == 1
    assert len(qa_lines) == 1

    paper = json.loads(paper_lines[0])
    qa = json.loads(qa_lines[0])

    assert paper["paper_id"] == "paper-a"
    assert [p["paragraph_id"] for p in paper["paragraphs"]] == [
        "paper-a::p0000",
        "paper-a::p0001",
    ]
    assert qa["question_id"] == "paper-a::q-a"
    assert qa["paper_id"] == "paper-a"
    assert qa["evidence_ids"] == ["paper-a::p0000"]
    assert qa["evidence_matches"] == [
        {
            "evidence": "Graph retrieval links terms in paper paragraphs.",
            "paragraph_id": "paper-a::p0000",
            "match_type": "exact",
        }
    ]
    assert qa["answers"] == ["Graph retrieval"]
    assert qa["unanswerable"] is False


def test_hf_columnar_answers_are_normalized():
    raw_records = [
        {
            "id": "paper-b",
            "full_text": [
                {
                    "section_name": "Intro",
                    "paragraphs": ["Graph retrieval links terms in paper paragraphs."],
                }
            ],
            "qas": {
                "question": ["What links terms?"],
                "question_id": ["q-b"],
                "answers": [
                    {
                        "answer": [
                            {
                                "extractive_spans": ["Graph retrieval"],
                                "free_form_answer": "",
                                "evidence": ["Graph retrieval links terms in paper paragraphs."],
                                "unanswerable": False,
                                "yes_no": None,
                            }
                        ],
                        "annotation_id": ["ann-1"],
                        "worker_id": ["worker-1"],
                    }
                ],
            },
        }
    ]

    _, qas = normalize_qasper_records(raw_records)

    assert len(qas) == 1
    assert qas[0]["answers"] == ["Graph retrieval"]
    assert qas[0]["evidence"] == ["Graph retrieval links terms in paper paragraphs."]
    assert qas[0]["evidence_ids"] == ["paper-b::p0000"]


def test_evidence_text_can_match_paragraph_substrings():
    raw_records = [
        {
            "id": "paper-c",
            "full_text": [
                {
                    "section_name": "Intro",
                    "paragraphs": ["The method uses GraphRAG expansion to improve evidence coverage."],
                }
            ],
            "qas": [
                {
                    "question": "What improves evidence coverage?",
                    "answers": [
                        {
                            "answer": {
                                "extractive_spans": ["GraphRAG expansion"],
                                "evidence": ["GraphRAG expansion to improve evidence coverage"],
                                "unanswerable": False,
                            }
                        }
                    ],
                }
            ],
        }
    ]

    _, qas = normalize_qasper_records(raw_records)

    assert qas[0]["evidence_ids"] == ["paper-c::p0000"]
    assert qas[0]["evidence_matches"][0]["match_type"] == "partial"
