# DataExcavate Agent Handoff

本文档是唯一入口。agent 读完应能理解项目全貌、已有代码、当前进度，并直接开始开发。

---

## 当前任务背景
用户正在推进 DataExcavate_Proj，一个面向数据挖掘课程中期检查的 Minimum Runnable GraphRAG Baseline。当前重点已经从“实现最小 baseline”转向“验证运行、整理交接文档、规范协作与提交历史”。

不要重新从零理解项目；先按下面的阅读顺序进入状态。

## 必读顺序
README.md
项目运行入口、环境配置、输出文件和当前范围。
docs/PROJECT_HANDOFF.md
面向新成员的中文项目状态说明。
包含 PRD、ADR、issue tracker、报告模板各自的作用。
CONTEXT.md
项目术语表。后续写文档和 issue 时沿用这里的语言。
.scratch/midterm-graphrag-baseline/PRD.md
最小 baseline 的范围来源。
docs/adr/0001-midterm-baseline-uses-lightweight-graph-retrieval.md
架构决策：中期 baseline 使用 TF-IDF seed retrieval + 一跳规则共现图扩展，不引入 Neo4j、dense embeddings 或 mandatory LLM。
.scratch/midterm-graphrag-baseline/issues/
7 个 issue 均已标记 completed，底部 comments 记录对应 commit 和完成内容。
docs/COLLABORATOR_HANDOFF.md
面向合作者的缺陷、优化方向、分支/commit/PR/issue 协作规范。
docs/项目中期进展报告模板.md
课程中期报告要求。报告需要 commit 统计、分支协作说明、目录结构、运行日志、指标表、失败案例和 AI 使用记录。
## 当前实现状态
最小 baseline 已实现，核心文件：

run_midterm.py：一键 pipeline。
src/data_loader.py：本地 JSON/JSONL 或 HuggingFace QASPER 加载。
src/preprocess.py：Processed QASPER Slice 标准化。
src/audit.py：数据审计。
src/retrieval.py：纯标准库 TF-IDF RAG baseline。
src/graph_rag.py：Rule-Based Co-Occurrence GraphRAG。
src/answering.py：抽取式回答与 INSUFFICIENT_EVIDENCE 拒答。
src/evaluate.py：Evidence Recall@5、Answer Token F1、Refusal Accuracy、latency 和 Failure Cases。
tests/：pytest-style 测试和无依赖 smoke suite。

## 一、项目是什么

**DataExcavate GraphRAG Baseline** — 面向数据挖掘课程中期检查的 Evidence-Constrained GraphRAG QA System。

- 输入：QASPER 论文数据集（本地 JSON/JSONL 或 HuggingFace `allenai/qasper`）
- 输出：数据审计报告、TF-IDF baseline 指标、GraphRAG 指标、失败案例
- 运行环境：CPU-only，不依赖 GPU / Neo4j / dense embeddings / LLM API
- 当前范围：最小可运行基线（Minimum Runnable GraphRAG Baseline），满足中期报告需要，不追求最优效果

---

## 二、系统架构

### 2.1 数据流

```
raw QASPER records
    │
    ▼
┌─────────────────────────┐
│  data_loader.py         │  加载本地 JSON/JSONL 或 HuggingFace QASPER
│  preprocess.py          │  标准化 → papers.jsonl + qas.jsonl
└─────────────────────────┘
    │
    ├──► audit.py ────────────► audit.json
    │
    ├──► retrieval.py ────────► baseline_predictions.json + baseline_metrics.json
    │       (TfidfRetriever)
    │       │
    │       ▼
    │    answering.py (抽取式回答 / INSUFFICIENT_EVIDENCE 拒答)
    │
    ├──► graph_rag.py ────────► graphrag_predictions.json + graphrag_metrics.json
    │       (GraphRagRetriever: TF-IDF seeds → 规则共现图一跳扩展 → graph bonus rerank)
    │       │
    │       ▼
    │    answering.py (同上)
    │
    └──► evaluate.py ─────────► failure_cases.json
```

### 2.2 模块边界（不可跨模块混用）

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据加载 | `src/data_loader.py` | 加载 QASPER 并构建切片 |
| 数据标准化 | `src/preprocess.py` | 原始记录 → 标准化 paper/QA JSON |
| 数据审计 | `src/audit.py` | 计算长度统计、证据缺失、不可回答比例 |
| 词法检索 | `src/retrieval.py` | TF-IDF 索引、top-k 检索（纯标准库） |
| 图检索 | `src/graph_rag.py` | 规则术语抽取、共现图构建、一跳扩展、rerank |
| 回答生成 | `src/answering.py` | 抽取式回答选择、拒答决策 |
| 评测 | `src/evaluate.py` | Recall@5、Token F1、Refusal Accuracy、失败案例 |
| 编排 | `run_midterm.py` | 串联完整 pipeline、写 artifacts |

### 2.3 唯一架构决策（ADR-0001）

中期 baseline 使用 **TF-IDF seed retrieval + 一跳规则共现图扩展**，不使用 Neo4j、dense embeddings 或 LLM 关系抽取。新增重依赖需先写 ADR。

---

## 三、源码地图

### 3.1 `src/data_loader.py` — 50 行

| 函数 | 签名 | 作用 |
|------|------|------|
| `load_qasper_records` | `(source, split) -> list[dict]` | 加载。传 `source` 读本地 JSON/JSONL/JSON dict；不传则通过 `datasets` 库下载 `allenai/qasper` |
| `build_processed_slice` | `(output_dir, source, split, max_papers, max_qas) -> (papers, qas)` | 加载→标准化→写 JSONL，返回标准化记录 |

### 3.2 `src/preprocess.py` — 210 行

| 函数 | 作用 |
|------|------|
| `normalize_qasper_records` | 原始记录 → `papers`（含 `paper_id/title/abstract/paragraphs`）+ `qas`（含 `question_id/paper_id/question/answers/evidence/evidence_ids/unanswerable`）。支持 QASPER v1（dict `full_text`）和 v2（list `full_text`） |
| `write_processed_slice` | 写 `papers.jsonl` 和 `qas.jsonl` |
| `read_jsonl` | 读 JSONL 回 `list[dict]` |

关键标识符生成规则：
- `paper_id`: `raw["id"]` 或 `raw["paper_id"]` 或 `"paper-{raw_index:04d}"`
- `paragraph_id`: `"{paper_id}::p{index:04d}"`
- `question_id`: `"{paper_id}::{raw_question_id}"`

### 3.3 `src/audit.py` — 85 行

| 函数 | 作用 |
|------|------|
| `audit_processed_slice` | 计算：文档/段落长度统计（min/max/mean）、长文档（>4000词）/长段落（>250词）计数及ID、证据缺失或不完整（有evidence文本但匹配不到paragraph_id）、不可回答问题比例 |
| `write_audit` | 写 `audit.json` |

### 3.4 `src/retrieval.py` — 122 行

核心类 `TfidfRetriever`：

| 方法 | 作用 |
|------|------|
| `from_papers(papers)` (classmethod) | 从 papers 构建 retriever，每个 paragraph 作为一个 document |
| `retrieve(query, paper_id, top_k)` | 返回 top-k 段落，每个段落附 `score`（cosine similarity） |
| `retrieve_with_latency(query, paper_id, top_k)` | 同上 + 返回耗时（ms） |
| `_cosine` | IDF 加权余弦相似度 |
| `_token_counts` | `re.findall(r"[A-Za-z0-9]+")` 分词并 lower |

`run_tfidf_baseline(papers, qas, top_k)` — 完整 TF-IDF pipeline：构建→检索→回答→写入预测。

### 3.5 `src/graph_rag.py` — 151 行

核心类 `GraphRagRetriever`（持有内部 TF-IDF retriever）：

| 方法 | 作用 |
|------|------|
| `from_papers(papers, tfidf)` | 构建 retriever，自动构建共现图 |
| `retrieve(query, paper_id, top_k, seed_k=2, graph_bonus=0.15)` | 两阶段检索：① TF-IDF 取 `seed_k` 种子段落 ② 从种子和 query 提取术语 ③ 通过共现图一跳扩展 ④ 收集所有含扩展术语的候选段落 ⑤ 按 `lexical_score + graph_bonus * graph_matches` 排序 |
| `has_query_match` | 检查检索结果段落是否与 query 有术语交集（用于拒答判断） |
| `_build_graph` | 对每个段落提取术语（`extract_terms`），构建 `term_paragraphs` 映射和 `neighbors` 共现关系 |

`extract_terms(text)` — 分词后保留 `len>=4` 且不在 `STOPWORDS` 中的 token。

`run_graph_rag(papers, qas, top_k)` — 完整 GraphRAG pipeline。

### 3.6 `src/answering.py` — 39 行

| 函数 | 作用 |
|------|------|
| `answer_from_evidence(question, evidence, min_score=0.05)` | 若 evidence 为空或 top score < 0.05 → 返回 `"INSUFFICIENT_EVIDENCE"`；否则从 top evidence 段落中选与问题 token overlap 最高的句子作为答案 |
| `_best_sentence` | 按 `(query token overlap, -sentence length)` 排序选最优句 |

常量 `INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"`。

### 3.7 `src/evaluate.py` — 136 行

| 函数 | 作用 |
|------|------|
| `evaluate_predictions(qas, predictions, top_k)` | 计算 4 个指标：Evidence Recall@5、Answer Token F1、Refusal Accuracy、Average Latency |
| `select_failure_cases(qas, predictions, limit)` | 选择最多 `limit` 个真实失败案例，分类为 `retrieval_miss` / `over_refusal` / `refusal_miss` / `answer_mismatch`，每个附 `failure_reason` 和 `improvement_direction` |
| `_best_answer_f1` | 预测答案与每个 reference answer 的 token-level F1 取最大值 |
| `write_json` | 通用 JSON 写入 |

### 3.8 `run_midterm.py` — 85 行（入口）

`run_pipeline(source, output_dir, max_papers, max_qas, top_k, split)` 串联全流程：

1. `build_processed_slice` → papers, qas
2. `audit_processed_slice` + `write_audit` → audit.json
3. `run_tfidf_baseline` + `evaluate_predictions` → baseline_predictions.json + baseline_metrics.json
4. `run_graph_rag` + `evaluate_predictions` → graphrag_predictions.json + graphrag_metrics.json
5. `select_failure_cases`（GraphRAG 优先，不足再从 baseline 补） → failure_cases.json

CLI 参数：`--source` `--output-dir` `--max-papers` `--max-qas` `--top-k` `--split`

---

## 四、输出文件格式

运行后 `results/midterm/` 下产生：

| 文件 | 格式 | 内容 |
|------|------|------|
| `processed/papers.jsonl` | 每行一个 JSON | `paper_id, title, abstract, paragraphs[{paragraph_id, paper_id, section, text}]` |
| `processed/qas.jsonl` | 每行一个 JSON | `question_id, paper_id, question, answers[], evidence[], evidence_ids[], unanswerable` |
| `audit.json` | JSON dict | `papers, qas, paragraphs, document_lengths{min,max,mean}, paragraph_lengths{min,max,mean}, long_documents{threshold,count,ids}, long_paragraphs{threshold,count,ids}, evidence{missing_or_incomplete_count,question_ids}, unanswerable{count,share,question_ids}` |
| `baseline_predictions.json` | JSON array | 每个 QA 一条：`question_id, paper_id, question, predicted_answer, retrieved_evidence_ids[], retrieved_evidence[], scores[], latency_ms, refused` |
| `baseline_metrics.json` | JSON dict | `questions, evidence_recall_at_5, answer_token_f1, refusal_accuracy, average_latency_ms` |
| `graphrag_predictions.json` | JSON array | 同上 + `graph_match` 字段 |
| `graphrag_metrics.json` | JSON dict | 同上 |
| `failure_cases.json` | JSON array | `question_id, question, prediction, reference_answers[], gold_evidence_ids[], retrieved_evidence_ids[], failure_reason, improvement_direction` |

---

## 五、运行与验证

### 5.1 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # 仅 datasets>=2.19, pytest>=8.0
```

### 5.2 一键运行

```bash
# 真实 QASPER（需网络）
python3 run_midterm.py --max-papers 20 --max-qas 60 --top-k 5 --output-dir results/midterm

# 本地离线 fixture
python3 run_midterm.py --source path/to/qasper.jsonl --max-papers 1 --max-qas 1 --output-dir results/smoke
```

### 5.3 测试

```bash
python3 -m pytest                    # 需要安装 pytest
python3 tests/run_smoke_tests.py     # 无 pytest 依赖的 smoke suite
```

所有测试使用本地 tiny fixture，不依赖网络和真实 QASPER。

---

## 六、当前完成状态

### 6.1 已完成（7/7 issues，全部 closed）

1. Processed QASPER Slice smoke path — tiny fixture 标准化到 JSONL
2. Midterm Dataset Slice and Data Audit — QASPER 加载、切片上限、审计
3. TF-IDF RAG Baseline — 检索 + 抽取式回答 + 指标
4. Rule-Based Co-Occurrence GraphRAG — 共现图 + 一跳扩展 + rerank
5. Retrieval-Based Refusal and Failure Cases — 拒答逻辑 + 失败案例
6. One-command midterm runner and README — `run_midterm.py` + README
7. Minimal test suite — pytest + smoke suite

### 6.2 明确不做的（PRD out of scope）

- Dense embedding / FAISS
- Neo4j 或图数据库持久化
- LLM-based entity/relation extraction
- 强制 LLM 回答生成
- RAGAS、NLI 或人工评测
- 全量 QASPER benchmark
- 高质量实体规范化、别名合并

---

## 七、开发规范

### 7.1 分支命名

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feature/` | 新功能 | `feature/bm25-retrieval` |
| `fix/` | bug 修复 | `fix/qasper-evidence-mapping` |
| `docs/` | 文档 | `docs/midterm-report-materials` |
| `experiment/` | 实验 | `experiment/dense-retrieval` |

### 7.2 Commit

- 至少一人一次有效 commit
- 用动宾结构：`Add BM25 retrieval baseline`
- 不提交 `data/processed/`、`results/`、`.venv`、`.env`、`*.log`

### 7.3 Issue 流程

1. `.scratch/` 下写 issue（标题、目标、验收标准、依赖）
2. 开分支实现
3. 完成后将 `Status:` 改为 `completed`，底部写对应 commit

### 7.4 添加新依赖

当前仅 `datasets` 和 `pytest`。新增依赖必须：
- 不破坏 CPU-only 承诺
- 不在必跑路径上引入 API key 依赖
- 更新 `requirements.txt`

---

## 八、如何开始开发

### 场景 A：修复 bug

1. 复现：先跑 `pytest` 或 `run_midterm.py` 确认 bug 存在
2. 在 `.scratch/` 新建 issue
3. 开 `fix/<bug-name>` 分支修改
4. 跑测试验证，提交 PR

### 场景 B：优化检索效果（P1-P2）

1. 确认不破坏现有 baseline 指标
2. 可做的改进：
   - 用 sklearn `TfidfVectorizer` / BM25 替换手写 TF-IDF
   - 对 graph expansion 增加 trace 输出（seed、expanded、bonus）
   - 增加 noun phrase / keyphrase extraction
   - 给 co-occurrence edge 加权
3. 新增模块或扩展现有模块，保持接口兼容
4. 跑全流程对比新旧指标

### 场景 C：添加 Optional LLM Enhancement（P3）

1. 保持默认不开启
2. 无 API key 时 baseline 仍完整运行
3. 输出中标记 LLM answer 与 extractive answer
4. 不混入默认 baseline 指标

### 场景 D：准备中期报告

1. 在统一环境运行真实 QASPER 小切片：`python3 run_midterm.py --max-papers 20 --max-qas 60`
2. 收集：`audit.json`、`baseline_metrics.json`、`graphrag_metrics.json`、`failure_cases.json`
3. 对照 `docs/项目中期进展报告模板.md` 填报告

---

## 九、关键文件清单

```
.
├── README.md                     # 项目说明与运行指南
├── AGENTS.md                     # Agent 配置（issue tracker / triage labels / domain docs）
├── CONTEXT.md                    # 领域术语表（13 个标准术语）
├── requirements.txt              # 依赖：datasets>=2.19, pytest>=8.0
├── run_midterm.py                # 一键入口
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── audit.py
│   ├── retrieval.py
│   ├── graph_rag.py
│   ├── answering.py
│   └── evaluate.py
├── tests/
│   ├── run_smoke_tests.py        # 无 pytest smoke suite
│   ├── test_runner_smoke.py
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   ├── test_audit.py
│   ├── test_tfidf_baseline.py
│   ├── test_graph_rag.py
│   └── test_refusal_failures.py
├── docs/
│   ├── adr/0001-*.md             # 唯一架构决策
│   ├── PROJECT_HANDOFF.md        # 面向新成员的状态说明
│   ├── COLLABORATOR_HANDOFF.md   # 缺陷、优化方向、协作规范
│   ├── agent_handoff.md          # 本文档
│   └── 项目中期进展报告模板.md   # 课程要求
└── .scratch/
    └── midterm-graphrag-baseline/
        ├── PRD.md                # 产品需求文档（36 user stories）
        └── issues/               # 7 个已完成 issue
```
