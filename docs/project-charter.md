# 智能合同审查 RAG Agent —— 项目章程

> 版本：1.0  
> 更新日期：2026-06-18  

---

## 1. 场景与核心功能

**场景**：智能合同审查系统（风险条款识别 + 跨合同比对）

**核心功能**：
- 合同上传与异步解析（PDF / Word → 条款级索引）
- 条款级问答（答案带原文引用）
- 风险条款自动检测（关键词匹配 + LLM 二次判断）
- 跨合同比对（差异分析 + 新增风险提示）

---

## 2. 技术栈清单

| 组件 | 方案 | 版本 / 备注 |
|------|------|-------------|
| API 框架 | FastAPI | ≥0.115 |
| Agent 编排 | LangGraph + LangChain | ≥0.2 / ≥0.3 |
| LLM | DeepSeek API | `deepseek-chat` |
| 向量数据库 | Qdrant（混合检索） | ≥1.12 |
| 文档解析 | Docling + Unstructured | 最新 |
| RAG 评估 | Ragas | ≥0.2 |
| 可观测性 | LangFuse Cloud | 最新 |
| 任务队列 | Celery + Redis | ≥5.4 / ≥7 |
| 主数据库 | PostgreSQL | ≥15 |
| 部署 | Docker Compose | 最新 |
| CI/CD & 代码质量 | GitHub Actions + Ruff + Pytest + pre-commit | 最新 |
| Embedding 模型 | BAAI/bge-small-zh-v1.5 | sentence-transformers 加载，102M 参数量 |

---

## 3. 开发里程碑与目录映射

| 里程碑 | 周期 | 核心产出 | 主要目录 |
|--------|------|----------|----------|
| 0. 初始化 | 0.5天 | 环境、docker-compose、配置 | 根目录：`pyproject.toml`, `.env`, `docker-compose.yml`, `scripts/` |
| 1. 上传 + 解析 | 2天 | 上传 → 解析 → 分块 → 索引 | `src/api/upload/`, `src/tasks/`, `src/parsers/`, `src/chunkers/`, `src/indexing/` |
| 2. 基础 RAG 问答 | 3天 | 单路检索 + 生成 | `src/graph/`（初版）, `src/retrieval/`（稠密向量） |
| 3. 核心优化 | 3-4天 | 混合检索 + 重排序 + 流式响应 | `src/retrieval/`（完整版）, `src/graph/nodes.py`（增加改写/回退） |
| 4. 特色功能 | 2-3天 | 风险检测 + 跨合同比对 | `src/risk/`, `src/comparator/`, `config/risk_rules.yaml` |
> M4 范围边界：
> - 风险检测与跨合同比对接口暂为同步模式，异步化改造（Celery）规划于 M5 工程化阶段实施
> - 不纳入本次开发（详见 `docs/PROGRESS.md`「项目反思与展望」）：版本去重、条款层级树、N≥3 比对基线、批量风险扫描、标准模板对比 |
| 5. 工程化 + 评估 | 2-3天 | CI、评估体系、可观测性 | `.github/workflows/`, `tests/eval/`, `src/utils/tracing.py` |

---

## 4. 关键文件路径（开发时常需定位）

- 环境变量模板：`.env.example`
- 风险规则配置：`config/risk_rules.yaml`
- FastAPI 入口：`src/api/main.py`
- Celery 应用定义：`src/tasks/celery_app.py`
- LangGraph 图构建：`src/graph/graph_builder.py`
- 测试评估脚本：`tests/eval/test_ragas.py`

---

## 5. RAG 优化手段落地索引

| 阶段 | 优化手段 | 实现位置 |
|------|----------|----------|
| 选型阶段 | 混合检索（稠密+稀疏）、结构化分块、可观测性、异步任务 | `vector_client.py`, `legal_chunker.py`, `tracing.py`, `tasks/` |
| 核心开发 | 查询改写、多路召回、重排序、置信度回退、流式响应、风险规则引擎、跨合同比对 | `retrieval/`, `graph/nodes.py`, `api/routes_qa.py`, `risk/`, `comparator/` |
| 工程化 | 语义缓存、任务监控、自动化评估、Bad Case 分析、分块调参 | `redis_client.py`, Celery Flower, `tests/eval/`, LangFuse, `config/chunk_experiments.yaml` |

---

## 6. 验收标准

- [x] 上传合同，异步解析并索引（Docling 29 页 PDF ~80s，Unstructured .doc ~22s）
- [x] 问答返回带条款编号的答案
- [x] 风险检测输出类型与原文引用
- [x] 跨合同比对输出差异表
- [x] Swagger 文档 `/docs` 可交互
- [~] Ragas 评估首次基线：Faithfulness 0.67（目标 0.80，差距来自 BGE-small 语义精度和 DeepSeek n=1 限制，详见 PROGRESS.md）
- [x] `docker compose up` 一键启动所有服务（含多阶段构建 + 健康检查）

---

## 7. 后续开发指引

- 本项目章程为技术决策的最终依据。
- 关于开发规范、交互流程和约束规则，请同时阅读根目录下的 `CLAUDE.md`。
- 当前进度请查阅 `docs/PROGRESS.md`。
- 遇到本章程未覆盖的实现细节，优先查阅各技术组件的官方文档，或与我确认后再继续。