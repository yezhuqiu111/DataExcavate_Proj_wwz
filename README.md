# DataExcavate GraphRAG Baseline

Minimum Runnable GraphRAG Baseline for evidence-constrained QA over QASPER-style computer science papers.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default code path is CPU-only. `datasets` is only needed when downloading real QASPER from HuggingFace. A local JSON/JSONL source can run offline.

## One-command Run

Real QASPER, small midterm slice:

```bash
python3 run_midterm.py --max-papers 20 --max-qas 60 --top-k 5 --output-dir results/midterm
```

Offline local fixture:

```bash
python3 run_midterm.py --source path/to/qasper.jsonl --max-papers 1 --max-qas 1 --output-dir results/smoke
```

## Outputs

Artifacts are written under the selected output directory:

- `processed/papers.jsonl`
- `processed/qas.jsonl`
- `audit.json`
- `baseline_predictions.json`
- `baseline_metrics.json`
- `graphrag_predictions.json`
- `graphrag_metrics.json`
- `failure_cases.json`

Generated `data/processed/` and `results/` outputs are ignored by git.

## Repo Structure

- `src/data_loader.py`: local/HuggingFace QASPER loading and slice creation
- `src/preprocess.py`: Processed QASPER Slice normalization
- `src/audit.py`: Data Audit Findings
- `src/retrieval.py`: TF-IDF RAG Baseline
- `src/graph_rag.py`: Rule-Based Co-Occurrence GraphRAG path
- `src/answering.py`: Evidence-Constrained Extractive Answer and refusal
- `src/evaluate.py`: metrics and Failure Case selection
- `run_midterm.py`: one-command runner
- `tests/`: tiny fixture behavior tests

Optional LLM Enhancement is not required for baseline reproduction.
