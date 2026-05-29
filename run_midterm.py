from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.audit import audit_processed_slice, write_audit
from src.data_loader import build_processed_slice
from src.evaluate import evaluate_predictions, select_failure_cases, write_json
from src.graph_rag import run_graph_rag
from src.retrieval import run_tfidf_baseline


def run_pipeline(
    source: str | Path | None = None,
    output_dir: str | Path = "results/midterm",
    max_papers: int = 20,
    max_qas: int = 60,
    top_k: int = 5,
    split: str = "train",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    processed_dir = output_path / "processed"
    papers, qas = build_processed_slice(
        output_dir=processed_dir,
        source=source,
        split=split,
        max_papers=max_papers,
        max_qas=max_qas,
    )

    audit = audit_processed_slice(papers, qas)
    write_audit(audit, output_path / "audit.json")

    baseline_predictions = run_tfidf_baseline(papers, qas, top_k=top_k)
    baseline_metrics = evaluate_predictions(qas, baseline_predictions, top_k=top_k)
    write_json(output_path / "baseline_predictions.json", baseline_predictions)
    write_json(output_path / "baseline_metrics.json", baseline_metrics)

    graphrag_predictions = run_graph_rag(papers, qas, top_k=top_k)
    graphrag_metrics = evaluate_predictions(qas, graphrag_predictions, top_k=top_k)
    write_json(output_path / "graphrag_predictions.json", graphrag_predictions)
    write_json(output_path / "graphrag_metrics.json", graphrag_metrics)

    failure_cases = select_failure_cases(qas, graphrag_predictions, limit=2)
    if len(failure_cases) < 2:
        failure_cases.extend(select_failure_cases(qas, baseline_predictions, limit=2 - len(failure_cases)))
    write_json(output_path / "failure_cases.json", failure_cases)

    return {
        "papers": len(papers),
        "qas": len(qas),
        "output_dir": str(output_path),
        "baseline": baseline_metrics,
        "graphrag": graphrag_metrics,
        "failure_cases": len(failure_cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Minimum Runnable GraphRAG Baseline.")
    parser.add_argument("--source", default=None, help="Optional local QASPER JSON/JSONL source. Omit to load allenai/qasper.")
    parser.add_argument("--output-dir", default="results/midterm", help="Artifact output directory.")
    parser.add_argument("--max-papers", type=int, default=20, help="Maximum papers in the Midterm Dataset Slice.")
    parser.add_argument("--max-qas", type=int, default=60, help="Maximum QA examples in the Midterm Dataset Slice.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence paragraphs to retrieve.")
    parser.add_argument("--split", default="train", help="QASPER split for HuggingFace datasets.")
    args = parser.parse_args()

    summary = run_pipeline(
        source=args.source,
        output_dir=args.output_dir,
        max_papers=args.max_papers,
        max_qas=args.max_qas,
        top_k=args.top_k,
        split=args.split,
    )
    print(f"papers={summary['papers']} qas={summary['qas']} output_dir={summary['output_dir']}")
    print(f"baseline_recall@5={summary['baseline']['evidence_recall_at_5']:.3f} baseline_f1={summary['baseline']['answer_token_f1']:.3f}")
    print(f"graphrag_recall@5={summary['graphrag']['evidence_recall_at_5']:.3f} graphrag_f1={summary['graphrag']['answer_token_f1']:.3f}")
    print(f"failure_cases={summary['failure_cases']}")


if __name__ == "__main__":
    main()
