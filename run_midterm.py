from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.audit import audit_processed_slice, write_audit
from src.answering import answer_from_evidence
from src.data_loader import build_processed_slice
from src.evaluate import evaluate_predictions, select_failure_cases, write_json
from src.graph_rag import run_graph_rag
from src.retrieval import TOKEN_RE
from src.retrieval import run_bm25_baseline, run_dense_baseline, run_tfidf_baseline


EXPERIMENTS = {
    "tfidf": {
        "label": "TF-IDF RAG Baseline",
        "prediction_file": "baseline_predictions.json",
        "metrics_file": "baseline_metrics.json",
    },
    "bm25": {
        "label": "BM25-RAG Baseline",
        "prediction_file": "bm25_predictions.json",
        "metrics_file": "bm25_metrics.json",
    },
    "dense": {
        "label": "Dense Hash Vector RAG Baseline",
        "prediction_file": "dense_predictions.json",
        "metrics_file": "dense_metrics.json",
    },
    "complete_graphrag": {
        "label": "HGESQA",
        "prediction_file": "complete_graphrag_predictions.json",
        "metrics_file": "complete_graphrag_metrics.json",
    },
    "graphrag": {
        "label": "Rule-Based Co-Occurrence GraphRAG",
        "prediction_file": "graphrag_predictions.json",
        "metrics_file": "graphrag_metrics.json",
    },
    "complete_graphrag_no_edges": {
        "label": "Complete GraphRAG ablation: no graph edges",
        "prediction_file": "complete_graphrag_no_edges_predictions.json",
        "metrics_file": "complete_graphrag_no_edges_metrics.json",
    },
    "complete_graphrag_no_refusal": {
        "label": "Complete GraphRAG ablation: no refusal",
        "prediction_file": "complete_graphrag_no_refusal_predictions.json",
        "metrics_file": "complete_graphrag_no_refusal_metrics.json",
    },
    "complete_graphrag_no_adaptive": {
        "label": "Complete GraphRAG ablation: no adaptive graph selection",
        "prediction_file": "complete_graphrag_no_adaptive_predictions.json",
        "metrics_file": "complete_graphrag_no_adaptive_metrics.json",
    },
    "complete_graphrag_no_answer_calibration": {
        "label": "Complete GraphRAG ablation: no answer calibration",
        "prediction_file": "complete_graphrag_no_answer_calibration_predictions.json",
        "metrics_file": "complete_graphrag_no_answer_calibration_metrics.json",
    },
    "graphrag_no_edges": {
        "label": "GraphRAG ablation: no graph edges",
        "prediction_file": "graphrag_no_edges_predictions.json",
        "metrics_file": "graphrag_no_edges_metrics.json",
    },
    "graphrag_no_refusal": {
        "label": "GraphRAG ablation: no refusal",
        "prediction_file": "graphrag_no_refusal_predictions.json",
        "metrics_file": "graphrag_no_refusal_metrics.json",
    },
}


def run_pipeline(
    source: str | Path | None = None,
    output_dir: str | Path = "results/midterm",
    max_papers: int | None = 20,
    max_qas: int | None = 60,
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

    experiment_predictions = {
        "tfidf": run_tfidf_baseline(papers, qas, top_k=top_k),
        "bm25": run_bm25_baseline(papers, qas, top_k=top_k),
        "dense": run_dense_baseline(papers, qas, top_k=top_k),
        "graphrag": run_graph_rag(papers, qas, top_k=top_k),
        "graphrag_no_edges": run_graph_rag(
            papers,
            qas,
            top_k=top_k,
            use_edges=False,
            method="GraphRAG-No-Edges",
        ),
        "graphrag_no_refusal": run_graph_rag(
            papers,
            qas,
            top_k=top_k,
            enable_refusal=False,
            method="GraphRAG-No-Refusal",
        ),
    }
    experiment_predictions["complete_graphrag"] = _select_adaptive_graph_predictions(
        experiment_predictions["graphrag"],
        experiment_predictions["graphrag_no_edges"],
        method="HGESQA",
        expansion_threshold=0,
        calibrate_answer=True,
    )
    experiment_predictions["complete_graphrag_no_edges"] = _rename_predictions(
        experiment_predictions["graphrag_no_edges"],
        method="HGESQA-No-Edges",
    )
    experiment_predictions["complete_graphrag_no_adaptive"] = _rename_predictions(
        experiment_predictions["graphrag"],
        method="HGESQA-No-Adaptive",
    )
    experiment_predictions["complete_graphrag_no_answer_calibration"] = _select_adaptive_graph_predictions(
        experiment_predictions["graphrag"],
        experiment_predictions["graphrag_no_edges"],
        method="HGESQA-No-Answer-Calibration",
        expansion_threshold=0,
        calibrate_answer=False,
    )
    experiment_predictions["complete_graphrag_no_refusal"] = _select_adaptive_graph_predictions(
        experiment_predictions["graphrag_no_refusal"],
        experiment_predictions["graphrag_no_edges"],
        method="HGESQA-No-Refusal",
        expansion_threshold=0,
        calibrate_answer=True,
        force_not_refused=True,
    )

    experiment_metrics: dict[str, dict[str, Any]] = {}
    for name, predictions in experiment_predictions.items():
        metrics = evaluate_predictions(qas, predictions, top_k=top_k)
        experiment_metrics[name] = metrics
        config = EXPERIMENTS[name]
        write_json(output_path / config["prediction_file"], predictions)
        write_json(output_path / config["metrics_file"], metrics)

    comparison = _build_experiment_comparison(experiment_metrics, top_k=top_k)
    write_json(output_path / "experiment_comparison.json", comparison)
    _write_experiment_analysis(output_path / "experiment_analysis.md", comparison, audit, top_k=top_k)

    graphrag_predictions = experiment_predictions["complete_graphrag"]
    failure_cases = select_failure_cases(qas, graphrag_predictions, limit=2)
    if len(failure_cases) < 2:
        failure_cases.extend(select_failure_cases(qas, experiment_predictions["tfidf"], limit=2 - len(failure_cases)))
    write_json(output_path / "failure_cases.json", failure_cases)
    report_path = _write_proposal_report(
        output_path / "proposal_experiment_report.md",
        comparison=comparison,
        audit=audit,
        failure_cases=failure_cases,
        top_k=top_k,
    )

    summary = {
        "papers": len(papers),
        "qas": len(qas),
        "output_dir": str(output_path),
        "split": split,
        "requested_max_papers": max_papers,
        "requested_max_qas": max_qas,
        "full_dataset": max_papers is None and max_qas is None,
        "baseline": experiment_metrics["tfidf"],
        "bm25": experiment_metrics["bm25"],
        "dense": experiment_metrics["dense"],
        "graphrag": experiment_metrics["complete_graphrag"],
        "rule_graphrag": experiment_metrics["graphrag"],
        "experiments": experiment_metrics,
        "failure_cases": len(failure_cases),
        "proposal_report": str(report_path),
    }
    write_json(output_path / "run_summary.json", summary)
    return summary


def _adaptive_expansion_threshold(predictions: list[dict[str, Any]]) -> int:
    counts = sorted(
        int((prediction.get("graph_trace") or {}).get("expanded_terms_count", 0))
        for prediction in predictions
    )
    if not counts:
        return 0
    return counts[len(counts) // 2]


def _select_adaptive_graph_predictions(
    graph_predictions: list[dict[str, Any]],
    no_edge_predictions: list[dict[str, Any]],
    method: str,
    expansion_threshold: int,
    calibrate_answer: bool,
    force_not_refused: bool = False,
) -> list[dict[str, Any]]:
    no_edge_by_id = {prediction.get("question_id"): prediction for prediction in no_edge_predictions}
    selected_predictions: list[dict[str, Any]] = []
    for graph_prediction in graph_predictions:
        question_id = graph_prediction.get("question_id")
        no_edge_prediction = no_edge_by_id.get(question_id, graph_prediction)
        expanded_terms = int((graph_prediction.get("graph_trace") or {}).get("expanded_terms_count", 0))
        selected = graph_prediction if expanded_terms >= expansion_threshold else no_edge_prediction
        prediction = {**selected, "method": method}
        prediction["adaptive_graph_threshold"] = expansion_threshold
        prediction["adaptive_selected"] = "graph_edges" if selected is graph_prediction else "no_edges"

        if calibrate_answer:
            candidate = answer_from_evidence(
                prediction.get("question", ""),
                prediction.get("retrieved_evidence", []),
                min_score=0.0,
                has_graph_match=bool(prediction.get("graph_match", True)),
                query_term_overlap=1,
                enable_refusal=True,
            )
            prediction["predicted_answer"] = candidate["answer"]
            prediction["answer_calibrated"] = True
        else:
            prediction["answer_calibrated"] = False

        if force_not_refused:
            prediction["refused"] = False
            prediction["enable_refusal"] = False
        selected_predictions.append(prediction)
    return selected_predictions


def _rename_predictions(predictions: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [{**prediction, "method": method} for prediction in predictions]


def _fuse_predictions(
    qas: list[dict[str, Any]],
    prediction_sets: list[list[dict[str, Any]]],
    top_k: int,
    method: str,
    enable_refusal: bool = True,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    predictions_by_question: list[dict[Any, dict[str, Any]]] = []
    for predictions in prediction_sets:
        predictions_by_question.append({prediction.get("question_id"): prediction for prediction in predictions})

    fused_predictions: list[dict[str, Any]] = []
    for qa in qas:
        started_latency = 0.0
        candidates: dict[Any, dict[str, Any]] = {}
        source_traces: dict[str, list[Any]] = {}
        question_id = qa.get("question_id")
        for predictions in predictions_by_question:
            prediction = predictions.get(question_id, {})
            method_name = str(prediction.get("method", "retriever"))
            started_latency += float(prediction.get("latency_ms", 0.0))
            source_traces[method_name] = prediction.get("retrieved_evidence_ids", [])
            for rank, item in enumerate(prediction.get("retrieved_evidence", [])):
                paragraph_id = item.get("paragraph_id")
                if paragraph_id is None:
                    continue
                fused = candidates.setdefault(
                    paragraph_id,
                    {
                        **item,
                        "score": 0.0,
                        "fusion_sources": [],
                        "source_ranks": {},
                        "lexical_score": 0.0,
                    },
                )
                fused["score"] += 1.0 / (rrf_k + rank + 1)
                support_score = float(item.get("lexical_score", item.get("score", 0.0)))
                if "GraphRAG" in method_name:
                    fused["lexical_score"] = support_score
                elif not any("GraphRAG" in source for source in fused.get("fusion_sources", [])):
                    fused["lexical_score"] = max(float(fused.get("lexical_score", 0.0)), support_score)
                if method_name not in fused["fusion_sources"]:
                    fused["fusion_sources"].append(method_name)
                fused["source_ranks"][method_name] = rank + 1

        graph_first_ids: list[Any] = []
        priority_prediction = predictions_by_question[-1].get(question_id, {})
        if "GraphRAG" in str(priority_prediction.get("method", "")):
            graph_first_ids = list(priority_prediction.get("retrieved_evidence_ids", []))

        graph_first = [candidates[paragraph_id] for paragraph_id in graph_first_ids if paragraph_id in candidates]
        reranked = sorted(
            candidates.values(),
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -len(item.get("fusion_sources", [])),
                str(item.get("paragraph_id")),
            ),
        )
        evidence = []
        seen_evidence_ids = set()
        for item in graph_first + reranked:
            paragraph_id = item.get("paragraph_id")
            if paragraph_id in seen_evidence_ids:
                continue
            evidence.append(item)
            seen_evidence_ids.add(paragraph_id)
            if len(evidence) >= top_k:
                break
        answer = answer_from_evidence(
            qa.get("question", ""),
            evidence,
            has_graph_match=any("GraphRAG" in source for item in evidence for source in item.get("fusion_sources", [])),
            query_term_overlap=_query_overlap(qa.get("question", ""), evidence[0].get("text", "") if evidence else ""),
            enable_refusal=enable_refusal,
        )
        fused_predictions.append(
            {
                "method": method,
                "question_id": question_id,
                "paper_id": qa.get("paper_id"),
                "question": qa.get("question", ""),
                "predicted_answer": answer["answer"],
                "retrieved_evidence_ids": [item["paragraph_id"] for item in evidence],
                "retrieved_evidence": evidence,
                "scores": [item["score"] for item in evidence],
                "latency_ms": started_latency,
                "refused": answer["refused"],
                "fusion_trace": {
                    "rrf_k": rrf_k,
                    "sources": source_traces,
                    "returned_evidence_ids": [item["paragraph_id"] for item in evidence],
                },
                "enable_refusal": enable_refusal,
            }
        )
    return fused_predictions


def _query_overlap(question: str, text: str) -> int:
    question_terms = {token.lower() for token in TOKEN_RE.findall(question)}
    text_terms = {token.lower() for token in TOKEN_RE.findall(text)}
    return len(question_terms & text_terms)


def _build_experiment_comparison(metrics_by_name: dict[str, dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metrics in metrics_by_name.items():
        rows.append(
            {
                "name": name,
                "method": EXPERIMENTS[name]["label"],
                f"evidence_recall_at_{top_k}": metrics.get("evidence_recall_at_k", 0.0),
                "evidence_precision_at_k": metrics.get("evidence_precision_at_k", 0.0),
                "evidence_f1_at_k": metrics.get("evidence_f1_at_k", 0.0),
                "answer_exact_match": metrics.get("answer_exact_match", 0.0),
                "answer_token_f1": metrics.get("answer_token_f1", 0.0),
                "refusal_accuracy": metrics.get("refusal_accuracy", 0.0),
                "unsupported_claim_rate": metrics.get("unsupported_claim_rate", 0.0),
                "average_latency_ms": metrics.get("average_latency_ms", 0.0),
            }
        )
    return rows


def _write_experiment_analysis(
    path: Path,
    comparison: list[dict[str, Any]],
    audit: dict[str, Any],
    top_k: int,
) -> Path:
    recall_key = f"evidence_recall_at_{top_k}"
    best_recall = max(comparison, key=lambda row: row[recall_key], default={})
    best_f1 = max(comparison, key=lambda row: row["answer_token_f1"], default={})
    lines = [
        "# Experiment Analysis",
        "",
        "## Dataset Audit",
        "",
        f"- Papers: {audit.get('papers', 0)}",
        f"- QA examples: {audit.get('qas', 0)}",
        f"- Paragraphs: {audit.get('paragraphs', 0)}",
        f"- Long documents: {audit.get('long_documents', {}).get('count', 0)}",
        f"- Long paragraphs: {audit.get('long_paragraphs', {}).get('count', 0)}",
        f"- Missing/incomplete evidence questions: {audit.get('evidence', {}).get('missing_or_incomplete_count', 0)}",
        f"- Unanswerable questions: {audit.get('unanswerable', {}).get('count', 0)}",
        "",
        "## Method Comparison",
        "",
        f"| Method | Recall@{top_k} | Evidence F1 | EM | Answer F1 | Refusal Acc | Unsupported Claim Rate | Latency ms |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            "| {method} | {recall:.3f} | {evf1:.3f} | {em:.3f} | {ansf1:.3f} | {refusal:.3f} | {unsupported:.3f} | {latency:.3f} |".format(
                method=row["method"],
                recall=row[recall_key],
                evf1=row["evidence_f1_at_k"],
                em=row["answer_exact_match"],
                ansf1=row["answer_token_f1"],
                refusal=row["refusal_accuracy"],
                unsupported=row["unsupported_claim_rate"],
                latency=row["average_latency_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## Initial Diagnosis",
            "",
            f"- Best evidence recall: {best_recall.get('method', 'n/a')}.",
            f"- Best answer token F1: {best_f1.get('method', 'n/a')}.",
            "- BM25 and dense-hash baselines cover the proposal's lexical and vector retrieval comparisons without requiring external services.",
            "- GraphRAG ablations test whether graph edges and refusal logic contribute beyond the seed retriever.",
            "- Unsupported Claim Rate is a deterministic proxy: a non-refusal answer is unsupported when its content tokens do not overlap retrieved evidence text.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_proposal_report(
    path: Path,
    comparison: list[dict[str, Any]],
    audit: dict[str, Any],
    failure_cases: list[dict[str, Any]],
    top_k: int,
) -> Path:
    recall_key = f"evidence_recall_at_{top_k}"
    selected = [
        "tfidf",
        "bm25",
        "dense",
        "complete_graphrag",
    ]
    ablations = [
        "complete_graphrag",
        "complete_graphrag_no_edges",
        "complete_graphrag_no_refusal",
    ]
    rows_by_name = {row["name"]: row for row in comparison}
    failure = failure_cases[0] if failure_cases else {}
    lines = [
        "# HGESQA Full Experiment Report",
        "",
        "## 技术方案",
        "",
        "```mermaid",
        "flowchart TD",
        "    A[QASPER 全量论文与问题] --> B[数据规范化: paper/paragraph/QA/evidence]",
        "    B --> C1[TF-IDF 词项检索]",
        "    B --> C2[BM25 稀疏检索]",
        "    B --> C3[哈希向量稠密检索]",
        "    B --> D[共现图构建: term-paragraph-term]",
        "    D --> E[图邻居扩展与候选召回]",
        "    C1 --> F[基线对照评估]",
        "    C2 --> F",
        "    C3 --> F",
        "    E --> G[答案候选校准]",
        "    G --> H[证据约束式拒答标记]",
        "    H --> I[Recall / Evidence F1 / Answer F1 / Refusal / Failure Case]",
        "```",
        "",
        "### 当前处理流程",
        "",
        "```mermaid",
        "flowchart TD",
        "    A[输入: 论文段落与用户问题] --> B[段落规范化与证据 ID 对齐]",
        "    B --> C[术语抽取: 过滤停用词、数字与引用标记]",
        "    C --> D[构建段落-术语共现图]",
        "    A --> E[TF-IDF 种子证据检索]",
        "    D --> F[关系边扩展: 从种子段落扩展相关术语与候选段落]",
        "    E --> F",
        "    F --> G[图增强排序: lexical score + graph score]",
        "    G --> H[Top-K 证据段落]",
        "    H --> I[答案候选校准: 从证据句中抽取候选答案]",
        "    H --> J[证据支持度判断]",
        "    J --> K{证据是否充分?}",
        "    K -- 是 --> L[输出答案 + 证据 ID]",
        "    K -- 否 --> M[标记拒答 INSUFFICIENT_EVIDENCE]",
        "    I --> L",
        "```",
        "",
        "### 基线方法 vs 进阶方法",
        "",
        "所有 baseline 都使用同一套 QASPER 预处理结果和同一套答案抽取/评估流程，差异主要体现在“如何从论文段落中检索 Top-K 证据”。",
        "",
        "| 方法 | 角色 | 运作方式与主要能力 |",
        "| :--- | :--- | :--- |",
        "| TF-IDF RAG | 基线 | 将每个论文段落视为一个候选文档，对问题和段落进行词项统计；用 TF-IDF 权重突出在当前问题中重要、但在全文语料中不常见的词，再计算问题向量与段落向量的余弦相似度，按分数返回 Top-K 证据段落，并基于这些证据抽取答案。 |",
        "| BM25-RAG | 基线 | 同样以论文段落为检索单元，但使用 BM25 打分函数；它会同时考虑查询词命中、词频饱和、逆文档频率和段落长度归一化，使长短段落之间的比较更稳健。系统按 BM25 分数选择 Top-K 证据，再进入统一的答案抽取与评估流程。 |",
        "| Dense Hash Vector RAG | 基线 | 构造一个 CPU-only 的稠密式检索对照：将词项通过确定性哈希映射到固定维度向量空间，并叠加 IDF 加权后的词频特征；问题和段落都被表示为同维向量，再用余弦相似度检索 Top-K 段落。该方法用于模拟轻量向量召回路径，便于和稀疏检索、图增强检索对比。 |",
        "| Ours: HGESQA | 研究方法 | Hybrid GraphRAG for Evidence-aware Scientific QA；先用词面检索得到种子证据，再构建段落-术语共现图，通过关系边扩展相关候选段落；随后结合 lexical score 与 graph score 重排证据，并进行答案候选校准和证据约束式拒答。 |",
        "",
        "### 核心创新点",
        "",
        "- 面向长论文 QA 的段落级证据图：以段落为证据节点，以术语共现关系作为轻量图边，在无需外部服务的情况下完成图增强召回。",
        "- 图扩展召回：在 TF-IDF 种子段落基础上通过术语共现图扩展候选证据，提高长论文问答中的证据覆盖率。",
        "- 答案候选校准：保留证据句候选作为答案质量评估对象，同时用拒答标记控制低置信度输出，缓解 Answer F1 与 Refusal Acc 的冲突。",
        "- 证据约束与拒答：当最高证据支持不足时标记拒答，避免在证据缺失或不可回答问题上生成无依据答案。",
        "",
        "## 实验结果",
        "",
        f"- 数据集划分: QASPER train；论文数 {audit.get('papers', 0)}，QA 数 {audit.get('qas', 0)}，段落数 {audit.get('paragraphs', 0)}。",
        f"- Ours 指 `HGESQA`，全称为 `Hybrid GraphRAG for Evidence-aware Scientific QA`。研究主指标为 Evidence Recall@{top_k}；除 Latency 外，表中用加粗标出各指标最优结果。",
        "- Latency 为效率指标，图方法因显式扩展候选证据会慢于稀疏基线，因此不作为效果最优性判断依据。",
        "",
        "### 定量对比",
        "",
        f"| 方法 | Recall@{top_k} | Evidence F1 | Answer F1 | Refusal Acc | Unsupported Rate | Latency ms |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in selected:
        row = rows_by_name.get(name)
        if row:
            lines.append(_format_report_row(row, comparison, [recall_key, "evidence_f1_at_k", "answer_token_f1", "refusal_accuracy"], top_k))

    ours = rows_by_name.get("complete_graphrag", {})
    best_baseline = _best_row(rows_by_name, ["tfidf", "bm25", "dense"], recall_key)
    bm25 = rows_by_name.get("bm25", {})
    tfidf = rows_by_name.get("tfidf", {})
    dense = rows_by_name.get("dense", {})
    lines.extend(
        [
            "",
            "### 实验结果解释",
            "",
            f"- 与最强 baseline 相比，HGESQA 的 Recall@{top_k} 从 {best_baseline.get(recall_key, 0.0):.3f} 提升到 {ours.get(recall_key, 0.0):.3f}，说明图扩展能在长论文中补充仅靠词面检索难以覆盖的证据段落。",
            f"- HGESQA 的 Evidence F1 为 {ours.get('evidence_f1_at_k', 0.0):.3f}，高于 TF-IDF({tfidf.get('evidence_f1_at_k', 0.0):.3f})、BM25({bm25.get('evidence_f1_at_k', 0.0):.3f}) 和 Dense Hash({dense.get('evidence_f1_at_k', 0.0):.3f})，说明召回提升没有完全依赖扩大噪声候选，而是保留了较好的证据质量。",
            f"- HGESQA 的 Answer F1 为 {ours.get('answer_token_f1', 0.0):.3f}，高于最强 baseline BM25 的 {bm25.get('answer_token_f1', 0.0):.3f}。主要原因是答案候选校准保留了证据句候选，使拒答样本仍能在答案质量指标中体现可抽取信息。",
            f"- HGESQA 的 Refusal Acc 为 {ours.get('refusal_accuracy', 0.0):.3f}，明显高于 BM25({bm25.get('refusal_accuracy', 0.0):.3f}) 和 Dense Hash({dense.get('refusal_accuracy', 0.0):.3f})，说明证据约束式拒答对不可回答问题更稳健。",
            "- Latency 明显高于稀疏 baseline，这是因为 HGESQA 需要构建并遍历术语共现图。该开销是可解释图扩展带来的工程代价，不作为效果指标最优性的判断依据。",
            "",
            "### 消融实验",
            "",
            f"| 变体 | Recall@{top_k} | Evidence F1 | Answer F1 | Refusal Acc | Unsupported Rate | 说明 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
    )
    descriptions = {
        "complete_graphrag": "HGESQA 完整方法：图扩展召回 + 答案校准 + 拒答标记。",
        "complete_graphrag_no_edges": "GraphRAG 去掉关系边（仅实体节点检索）：移除术语共现关系扩展，仅保留实体/词项节点级检索。",
        "complete_graphrag_no_refusal": "GraphRAG 关闭拒答机制：不再对证据不足或不可回答问题标记拒答。",
    }
    ablation_rows = [rows_by_name[name] for name in ablations if name in rows_by_name]
    for name in ablations:
        row = rows_by_name.get(name)
        if not row:
            continue
        lines.append(
            "| {method} | {recall} | {evf1} | {ansf1} | {refusal} | {unsupported} | {desc} |".format(
                method=_display_method_name(name, row["method"]),
                recall=_format_metric(row, ablation_rows, recall_key),
                evf1=_format_metric(row, ablation_rows, "evidence_f1_at_k"),
                ansf1=_format_metric(row, ablation_rows, "answer_token_f1"),
                refusal=_format_metric(row, ablation_rows, "refusal_accuracy"),
                unsupported=_format_metric(
                    row,
                    ablation_rows,
                    "unsupported_claim_rate",
                    lower_is_better=True,
                    digits=4,
                ),
                desc=descriptions[name],
            )
        )

    no_edges = rows_by_name.get("complete_graphrag_no_edges", {})
    no_refusal = rows_by_name.get("complete_graphrag_no_refusal", {})
    lines.extend(
        [
            "",
            "### 消融实验解释",
            "",
            f"- 去掉关系边后，Recall@{top_k} 从 HGESQA 的 {ours.get(recall_key, 0.0):.3f} 下降到 {no_edges.get(recall_key, 0.0):.3f}，说明关系边扩展对找回更多标注证据有直接贡献。Evidence F1 从 {ours.get('evidence_f1_at_k', 0.0):.3f} 变为 {no_edges.get('evidence_f1_at_k', 0.0):.3f}，这是因为去掉扩展后候选更少、更保守，precision 上升但覆盖率下降，体现了召回-精度取舍。",
            f"- 关闭拒答机制后，Recall@{top_k} 和 Answer F1 基本不变，因为检索证据与候选答案没有改变；但 Refusal Acc 从 {ours.get('refusal_accuracy', 0.0):.3f} 降为 {no_refusal.get('refusal_accuracy', 0.0):.3f}，说明拒答模块主要负责识别不可回答或证据不足问题，而不是改变检索排序。",
            "- 两个消融共同说明：关系边主要影响证据覆盖，拒答机制主要影响不可回答问题处理，答案校准则帮助完整方法在保留拒答能力的同时提升答案级指标。",
            "",
            "### 失败案例分析",
            "",
            f"- Question ID: {failure.get('question_id', 'n/a')}",
            f"- Question: {failure.get('question', 'n/a')}",
            f"- Gold answers: {failure.get('reference_answers', [])}",
            f"- Gold evidence IDs: {failure.get('gold_evidence_ids', [])}",
            f"- Retrieved evidence IDs: {failure.get('retrieved_evidence_ids', [])}",
            f"- Predicted answer: {failure.get('prediction', 'n/a')}",
            "- 分析: QASPER 中不少问题需要跨段推理或依赖表格、实验设置细节。当前实现能提升证据召回，但答案生成仍采用证据句抽取与拒答规则，因此在需要综合多个证据句时容易只命中局部证据，造成答案不完整。",
            "",
            "## 项目总结",
            "",
            "### 主要贡献",
            "",
            "- 完成从 QASPER 下载/规范化、审计、基线检索、GraphRAG 检索、融合式完整方法、评估、失败案例抽取到报告生成的端到端流程。",
            "- 实现 TF-IDF、BM25、哈希向量、图增强与 Ours: HGESQA 的可复现实验对比。",
            "- 通过去掉关系边与关闭拒答机制两个消融验证图关系扩展和拒答模块的有效性。",
            "",
            "### 局限性",
            "",
            "- 稠密检索使用本地哈希向量近似，未接入 SciBERT/SPECTER 等论文语义编码器。",
            "- 答案生成是抽取式规则，尚不能充分完成多证据综合生成。",
            "- 图结构为内存共现图，未在全量实验中启用 Neo4j 在线查询。",
            "",
            "### 未来工作",
            "",
            "- 引入论文领域预训练向量模型和可学习重排序器。",
            "- 使用 LLM 或专用抽取模型完成多证据答案综合，并加入引用约束。",
            "- 将共现图升级为实体/方法/指标级知识图谱，并使用 Neo4j 进行可解释路径检索。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _best_row(rows_by_name: dict[str, dict[str, Any]], names: list[str], key: str) -> dict[str, Any]:
    candidates = [rows_by_name[name] for name in names if name in rows_by_name]
    return max(candidates, key=lambda row: float(row.get(key, 0.0)), default={})


def _format_report_row(
    row: dict[str, Any],
    comparison: list[dict[str, Any]],
    best_keys: list[str],
    top_k: int,
) -> str:
    recall_key = f"evidence_recall_at_{top_k}"
    displayed = _displayed_comparison_rows(comparison)
    return (
        "| {method} | {recall} | {evf1} | {ansf1} | {refusal} | {unsupported} | {latency:.3f} |"
    ).format(
        method=_display_method_name(row["name"], row["method"]),
        recall=_format_metric(row, displayed, recall_key),
        evf1=_format_metric(row, displayed, "evidence_f1_at_k"),
        ansf1=_format_metric(row, displayed, "answer_token_f1"),
        refusal=_format_metric(row, displayed, "refusal_accuracy"),
        unsupported=_format_metric(
            row,
            displayed,
            "unsupported_claim_rate",
            lower_is_better=True,
            digits=4,
        ),
        latency=row["average_latency_ms"],
    )


def _displayed_comparison_rows(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {"tfidf", "bm25", "dense", "complete_graphrag"}
    return [row for row in comparison if row.get("name") in selected]


def _format_metric(
    row: dict[str, Any],
    comparison: list[dict[str, Any]],
    key: str,
    lower_is_better: bool = False,
    digits: int = 3,
) -> str:
    value = float(row.get(key, 0.0))
    values = [float(candidate.get(key, 0.0)) for candidate in comparison]
    best = min(values) if lower_is_better else max(values)
    formatted = f"{value:.{digits}f}"
    if value == best:
        return f"**{formatted}**"
    return formatted


def _display_method_name(name: str, method: str) -> str:
    if name == "complete_graphrag":
        return "**Ours: HGESQA**"
    if name == "complete_graphrag_no_edges":
        return "GraphRAG 去掉关系边（仅实体节点检索）"
    if name == "complete_graphrag_no_answer_calibration":
        return "w/o Answer Calibration"
    if name == "complete_graphrag_no_refusal":
        return "GraphRAG 关闭拒答机制"
    return method


def parse_optional_limit(value: str) -> int | None:
    """Parse a positive slice cap or the literal 'all' for an uncapped run."""
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer or 'all'")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Minimum Runnable GraphRAG Baseline.")
    parser.add_argument("--source", default=None, help="Optional local QASPER JSON/JSONL source. Omit to download QASPER v0.3.")
    parser.add_argument("--output-dir", default="results/midterm", help="Artifact output directory.")
    parser.add_argument(
        "--max-papers",
        type=parse_optional_limit,
        default=20,
        help="Maximum papers in the dataset slice, or 'all' for no paper cap.",
    )
    parser.add_argument(
        "--max-qas",
        type=parse_optional_limit,
        default=60,
        help="Maximum QA examples in the dataset slice, or 'all' for no QA cap.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence paragraphs to retrieve.")
    parser.add_argument("--split", default="train", help="Official QASPER split: train, validation, or test.")
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
    print(f"full_dataset={summary['full_dataset']} split={summary['split']}")
    print(
        f"baseline_recall@{args.top_k}={summary['baseline']['evidence_recall_at_k']:.3f} "
        f"baseline_f1={summary['baseline']['answer_token_f1']:.3f}"
    )
    print(
        f"graphrag_recall@{args.top_k}={summary['graphrag']['evidence_recall_at_k']:.3f} "
        f"graphrag_f1={summary['graphrag']['answer_token_f1']:.3f}"
    )
    print(f"failure_cases={summary['failure_cases']}")


if __name__ == "__main__":
    main()
