# DataExcavate GraphRAG Baseline

这是一个面向数据挖掘课程中期检查的最小可运行 GraphRAG baseline。项目目标不是追求最优模型效果，而是在 CPU 环境下跑通一个可复现的 Evidence-Constrained GraphRAG QA System：加载 QASPER 小切片，完成数据审计，运行 TF-IDF RAG baseline 和规则共现图增强检索，输出评测指标和失败案例，支撑中期报告填写。

## 当前状态

最小 baseline 已完成。当前代码支持：

- 读取本地 QASPER-like JSON/JSONL，或通过 HuggingFace `datasets` 下载真实 QASPER。
- 将原始数据标准化为 Processed QASPER Slice。
- 输出 `papers.jsonl` 和 `qas.jsonl`。
- 统计数据审计结果，包括文档/段落长度、证据缺失、不完整证据、不可回答问题比例。
- 运行 TF-IDF RAG Baseline。
- 运行 Rule-Based Co-Occurrence GraphRAG。
- 生成 Evidence-Constrained Extractive Answer。
- 触发 Retrieval-Based Refusal。
- 计算 Evidence Recall@5、Answer Token F1、Refusal Accuracy 和平均延迟。
- 导出真实评测结果中的 Failure Cases。

## 环境配置

推荐 Python 3.10+。项目核心逻辑尽量保持轻依赖；真实 QASPER 下载需要 `datasets`，测试需要 `pytest`。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认路径为 CPU-only，不需要 GPU、Neo4j、向量数据库、LLM API key 或大模型下载。

## 一键运行

使用真实 QASPER 小切片：

```bash
python3 run_midterm.py --max-papers 20 --max-qas 60 --top-k 5 --output-dir results/midterm
```

使用本地离线 JSONL fixture：

```bash
python3 run_midterm.py --source path/to/qasper.jsonl --max-papers 1 --max-qas 1 --top-k 5 --output-dir results/smoke
```

常用参数：

- `--source`：本地 QASPER-like JSON 或 JSONL 文件。不传时尝试通过 HuggingFace 加载 `allenai/qasper`。
- `--max-papers`：最多处理多少篇论文，默认 20。
- `--max-qas`：最多处理多少个 QA 样本，默认 60。
- `--top-k`：每个问题保留多少条证据段落，默认 5。
- `--output-dir`：结果输出目录，默认 `results/midterm`。
- `--split`：HuggingFace QASPER split，默认 `train`。

## 测试

安装依赖后运行：

```bash
python3 -m pytest
```

如果当前环境没有安装 `pytest`，可以先运行无 pytest 依赖的 smoke suite：

```bash
python3 tests/run_smoke_tests.py
```

测试使用本地小 fixture，不依赖网络，也不使用真实 QASPER 作为测试数据。

## 输出文件

运行后，输出目录下会生成：

- `processed/papers.jsonl`：标准化后的论文与段落记录。
- `processed/qas.jsonl`：标准化后的问题、答案和证据记录。
- `audit.json`：数据审计结果。
- `baseline_predictions.json`：TF-IDF RAG baseline 预测结果。
- `baseline_metrics.json`：TF-IDF RAG baseline 指标。
- `graphrag_predictions.json`：GraphRAG 路径预测结果。
- `graphrag_metrics.json`：GraphRAG 路径指标。
- `failure_cases.json`：可写入中期报告的失败案例。

`data/raw/`、`data/processed/` 和 `results/` 已在 `.gitignore` 中排除，不应提交生成数据和结果。

## 仓库结构

```text
.
├── README.md
├── AGENTS.md
├── CONTEXT.md
├── requirements.txt
├── run_midterm.py
├── src/
│   ├── data_loader.py       # QASPER 本地/远程加载与切片构建
│   ├── preprocess.py        # Processed QASPER Slice 标准化
│   ├── audit.py             # 数据审计
│   ├── retrieval.py         # TF-IDF RAG Baseline
│   ├── graph_rag.py         # 规则共现图 GraphRAG 路径
│   ├── answering.py         # 抽取式回答与拒答
│   └── evaluate.py          # 指标计算与失败案例选择
├── tests/
│   ├── run_smoke_tests.py
│   └── test_*.py
├── docs/
│   ├── adr/
│   ├── agents/
│   └── 项目中期进展报告模板.md
└── .scratch/
    └── midterm-graphrag-baseline/
        ├── PRD.md
        └── issues/
```

## 文档入口

新成员建议先读：

1. `docs/PROJECT_HANDOFF.md`
2. `README.md`
3. `CONTEXT.md`
4. `.scratch/midterm-graphrag-baseline/PRD.md`
5. `docs/adr/0001-midterm-baseline-uses-lightweight-graph-retrieval.md`
6. `.scratch/midterm-graphrag-baseline/issues/`
7. `docs/COLLABORATOR_HANDOFF.md`

## 范围说明

当前 baseline 明确不包含：

- Dense embedding / FAISS。
- Neo4j 或图数据库持久化。
- LLM-based entity/relation extraction。
- 强制 LLM 回答生成。
- RAGAS、NLI 或人工评测。
- 全量 QASPER benchmark。
- 高质量实体规范化、别名合并或性能优化。

Optional LLM Enhancement 可以作为后续方向，但不能阻塞当前 baseline 的复现。
