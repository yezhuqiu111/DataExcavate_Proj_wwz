# 项目当前状况与阅读交接

本文档给新同学或后续 agent 一个最短上手路径：先读什么、为什么读、当前进度到哪里、每类项目文档分别负责什么。

## 结论

当前项目已经完成中期所需的 Minimum Runnable GraphRAG Baseline。它可以从 QASPER 或 QASPER-like 本地文件构建小规模数据切片，输出数据审计、TF-IDF baseline、规则共现图 GraphRAG、评测指标和失败案例。

当前目标不是最优效果，而是满足课程中期检查对“数据预处理与审计、基线模型、核心算法进度、评测输出、一键复现”的要求。

## 推荐阅读顺序

### 1. `README.md`

先看 README，理解项目是什么、怎么安装、怎么运行、会输出哪些文件。README 是复现入口，不负责解释所有设计取舍。

### 2. `CONTEXT.md`

再看领域词汇表。这里定义了项目内部统一用语，例如：

- Minimum Runnable GraphRAG Baseline
- Midterm Dataset Slice
- Processed QASPER Slice
- TF-IDF RAG Baseline
- Rule-Based Co-Occurrence Graph
- Evidence-Constrained Extractive Answer
- Retrieval-Based Refusal
- Failure Case

后续写报告、issue、代码注释和 README 时应尽量沿用这些词，避免把当前 baseline 说成完整研究系统或生产系统。

### 3. `.scratch/midterm-graphrag-baseline/PRD.md`

PRD 是产品和实现范围的来源。它说明：

- 为什么要做这个最小 baseline。
- 用户故事是什么。
- 必须产出哪些 artifact。
- 哪些技术明确不做。
- 模块边界如何划分。
- 测试应该覆盖哪些外部行为。

后续如果要扩展功能，应先确认扩展是否仍符合 PRD；如果范围明显变化，应新建 PRD 或补充 issue。

### 4. `docs/adr/0001-midterm-baseline-uses-lightweight-graph-retrieval.md`

ADR 是架构决策记录。当前只有一个核心决策：中期 baseline 使用 TF-IDF seed retrieval 加一跳 Rule-Based Co-Occurrence Graph expansion，而不是 Neo4j、dense embeddings 或 LLM relation extraction。

这个文件的作用是约束后续改动：如果只是为了中期 baseline，不应引入重依赖、GPU、大模型或图数据库。若后续确实要改为 dense retrieval 或 Neo4j，应新增 ADR 解释为什么改变。

### 5. `.scratch/midterm-graphrag-baseline/issues/`

issues 是任务拆分和进度记录。当前 7 个 issue 都已标记为 `completed`：

1. `01-processed-qasper-slice-smoke-path.md`：完成 tiny fixture 到 processed JSONL 的最小数据契约。
2. `02-midterm-dataset-slice-data-audit-findings.md`：完成 QASPER 加载、切片上限和数据审计。
3. `03-tfidf-rag-baseline-end-to-end.md`：完成 TF-IDF 检索、抽取式回答和 baseline 指标。
4. `04-rule-based-co-occurrence-graphrag-path.md`：完成规则共现图、一跳扩展和 graph bonus rerank。
5. `05-retrieval-based-refusal-failure-cases.md`：完成拒答逻辑、Refusal Accuracy 和失败案例导出。
6. `06-one-command-midterm-runner-readme.md`：完成 `run_midterm.py` 和 README。
7. `07-minimal-test-suite-baseline-confidence.md`：完成 pytest-style 测试和无 pytest smoke suite。

每个 issue 底部都有 triage comment，记录对应 commit 和完成内容。

### 6. `docs/项目中期进展报告模板.md`

这是课程提交模板，不是代码文档。它告诉我们中期报告必须覆盖：

- 项目当前状态。
- 代码仓库状态和 commit 记录。
- 分支与协作方式。
- 实际目录结构。
- 数据审计问题和量化规模。
- 一键复现命令。
- 基线模型运行结果。
- 进阶算法开发进度。
- 指标对比表。
- 至少 2 个真实失败案例。
- 后续排期。
- AI 工具使用记录。

当前代码已经能生成其中多数需要的数据，但报告本身还需要人工整理、截图、填成员分工和粘贴真实运行日志。

### 7. `docs/COLLABORATOR_HANDOFF.md`

最后读合作者交接文档。它面向后续继续开发的人，列出当前 baseline 的缺陷、优化方向、协作规范和建议分支策略。

## 当前代码模块

- `src/data_loader.py`：加载本地 JSON/JSONL 或 HuggingFace QASPER，并构建 Midterm Dataset Slice。
- `src/preprocess.py`：把 QASPER-like 原始结构转换为标准 paper 和 QA JSONL。
- `src/audit.py`：计算文档长度、段落长度、证据缺失和不可回答比例。
- `src/retrieval.py`：纯标准库 TF-IDF 检索 baseline。
- `src/graph_rag.py`：规则 term 抽取、共现图构建、一跳扩展和 rerank。
- `src/answering.py`：从检索证据中选择句子回答，或输出 `INSUFFICIENT_EVIDENCE`。
- `src/evaluate.py`：计算 Evidence Recall@5、Answer Token F1、Refusal Accuracy、平均延迟，并选择 Failure Cases。
- `run_midterm.py`：串起完整流程并写出 report-ready artifacts。
- `tests/`：小 fixture 行为测试，不依赖真实 QASPER。

## 当前进度

已完成：

- 最小数据处理链路。
- 数据审计输出。
- TF-IDF RAG baseline。
- 规则共现图 GraphRAG 路径。
- 抽取式回答与拒答。
- 指标和失败案例导出。
- 一键 runner。
- README。
- 最小测试套件。

仍需人工处理：

- 在服务器或统一环境中跑真实 QASPER 小切片。
- 将运行日志、指标、失败案例整理进中期报告。
- 截取 GitHub commit history、目录结构和运行结果。
- 明确组员分工、后续排期和 AI 工具使用记录。

## 对后续 agent 的工作方式建议

1. 先读本文档和 README，不要直接改代码。
2. 新需求先查 PRD、ADR 和现有 issue，确认是否属于当前 baseline 范围。
3. 若是 bug 或优化，先新建或更新 `.scratch/` issue，再开分支实现。
4. 只提交代码、测试和必要文档；不要提交 `data/processed/`、`results/`、`.env` 或缓存。
5. 真实运行结果应在服务器或统一环境生成，避免不同本地环境造成报告数字混乱。
