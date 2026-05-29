import json

from src.data_loader import load_qasper_records
from src.preprocess import normalize_qasper_records


def test_local_qasper_jsonl_loader_supports_midterm_slice_caps(tmp_path):
    raw_path = tmp_path / "qasper.jsonl"
    rows = [
        {"id": "p1", "full_text": [], "qas": [{"question": "q1"}]},
        {"id": "p2", "full_text": [], "qas": [{"question": "q2"}]},
    ]
    raw_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = load_qasper_records(raw_path)
    papers, qas = normalize_qasper_records(records, max_papers=1, max_qas=1)

    assert [paper["paper_id"] for paper in papers] == ["p1"]
    assert len(qas) == 1
