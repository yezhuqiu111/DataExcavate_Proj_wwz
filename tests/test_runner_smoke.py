import json

from run_midterm import parse_optional_limit, run_pipeline
from run_scale_experiments import run_scale_experiments


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
    assert (tmp_path / "out" / "bm25_metrics.json").exists()
    assert (tmp_path / "out" / "dense_metrics.json").exists()
    assert (tmp_path / "out" / "graphrag_metrics.json").exists()
    assert (tmp_path / "out" / "graphrag_no_edges_metrics.json").exists()
    assert (tmp_path / "out" / "graphrag_no_refusal_metrics.json").exists()
    assert (tmp_path / "out" / "experiment_comparison.json").exists()
    assert (tmp_path / "out" / "experiment_analysis.md").exists()
    assert (tmp_path / "out" / "proposal_experiment_report.md").exists()
    assert (tmp_path / "out" / "failure_cases.json").exists()
    assert (tmp_path / "out" / "run_summary.json").exists()
    assert "bm25" in summary["experiments"]
    assert "dense" in summary["experiments"]


def test_runner_supports_uncapped_local_run_and_scale_comparison(tmp_path):
    source = tmp_path / "fixture.jsonl"
    rows = [
        {"id": "p1", "full_text": [], "qas": [{"question": "q1"}]},
        {"id": "p2", "full_text": [], "qas": [{"question": "q2"}]},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert parse_optional_limit("all") is None
    assert parse_optional_limit("3") == 3

    summary = run_pipeline(source=source, output_dir=tmp_path / "full", max_papers=None, max_qas=None)
    comparisons = run_scale_experiments(
        scales=[(1, 1), (None, None)],
        source=source,
        output_dir=tmp_path / "scales",
    )

    assert summary["full_dataset"]
    assert summary["papers"] == 2
    assert summary["qas"] == 2
    assert [result["qas"] for result in comparisons] == [1, 2]
    assert (tmp_path / "scales" / "comparison.json").exists()
