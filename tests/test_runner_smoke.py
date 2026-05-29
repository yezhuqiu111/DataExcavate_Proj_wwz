import json

from run_midterm import run_pipeline


def test_runner_writes_midterm_artifacts_from_local_fixture(tmp_path):
    source = tmp_path / "fixture.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "p1",
                "title": "Graph retrieval",
                "full_text": [
                    {
                        "section_name": "Intro",
                        "paragraphs": [
                            "Graph retrieval connects evidence terms.",
                            "Unrelated optimizer text.",
                        ],
                    }
                ],
                "qas": [
                    {
                        "question_id": "q1",
                        "question": "What connects evidence terms?",
                        "answers": [
                            {
                                "answer": {
                                    "extractive_spans": ["Graph retrieval"],
                                    "evidence": ["Graph retrieval connects evidence terms."],
                                    "unanswerable": False,
                                }
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_pipeline(source=source, output_dir=tmp_path / "out", max_papers=1, max_qas=1, top_k=2)

    assert summary["qas"] == 1
    assert (tmp_path / "out" / "processed" / "papers.jsonl").exists()
    assert (tmp_path / "out" / "audit.json").exists()
    assert (tmp_path / "out" / "baseline_metrics.json").exists()
    assert (tmp_path / "out" / "graphrag_metrics.json").exists()
    assert (tmp_path / "out" / "failure_cases.json").exists()
