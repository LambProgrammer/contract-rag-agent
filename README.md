# 智能合同审查 RAG Agent

基于 RAG（检索增强生成）技术的智能合同审查系统。上传合同 → 条款级索引 → 自然语言问答 → 风险自动检测 → 跨合同比对。

## 技术栈

| 领域 | 选型 |
|------|------|
| API 框架 | FastAPI |
| Agent 编排 | LangGraph + LangChain |
| LLM | DeepSeek API |
| 向量数据库 | Qdrant（混合检索：稠密 + 稀疏 + RRF 融合 + CrossEncoder 重排序） |
| 文档解析 | Docling（主力）+ Unstructured（回退） |
| Embedding 模型 | BAAI/bge-small-zh-v1.5（512 维） |
| 任务队列 | Celery + Redis |
| 数据库 | PostgreSQL 15 |
| 可观测性 | LangFuse Cloud（OpenTelemetry 追踪） |
| 评估 | Ragas（Faithfulness / AnswerRelevancy / ContextRecall / ContextRelevance） |
| CI/CD | GitHub Actions（ci.yml + cd.yml） |

## 快速开始

### 前置条件

- Docker Desktop
- DeepSeek API Key

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 2. 一键启动

```bash
docker compose up -d --build
```

### 3. 打开浏览器

http://localhost:8000

### 4. 本地开发模式（不构建 Docker）

```bash
# Docker Desktop 启动 postgres + redis + qdrant
docker compose up -d postgres redis qdrant

# 终端 1 — FastAPI
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 终端 2 — Celery Worker
uv run celery -A src.tasks.celery_app worker --loglevel=info --pool=solo
```

## RAG 管线

```
用户提问 → 查询改写 → 混合检索(dense+sparse) → 重排序 → 门控检查 → LLM 生成 → SSE 流式返回
                        ↑ 预检不通过 → fallback
```

## 评估基线

| 指标 | 分数 | 说明 |
|------|:--:|------|
| Faithfulness（忠实度） | 0.67 | 答案是否基于合同原文 |
| AnswerRelevancy（答案相关性） | 0.57 | 答案是否切中问题 |
| ContextRecall（召回率） | 0.18 | 检索是否覆盖所有相关条款 |
| ContextRelevance（上下文相关性） | 0.71 | 检索到的条款是否与问题相关 |

> 评估工具：Ragas 0.4.3，测试集 14 条（6 类场景），分数上报 LangFuse。

## 项目结构

```
contract-rag-agent/
├── src/
│   ├── api/            # FastAPI 路由（upload / qa / risk / compare）
│   ├── graph/          # LangGraph RAG 图（6 节点 + 条件边）
│   ├── retrieval/      # 混合检索 + 稀疏检索 + 重排序
│   ├── indexing/       # Embedding 向量化 + Qdrant 客户端
│   ├── parsers/        # 文档解析（Docling + Unstructured 回退）
│   ├── chunkers/       # 法律条款级分块
│   ├── risk/           # 风险检测（规则引擎 + LLM 二次确认）
│   ├── comparator/     # 跨合同比对（条款对齐 + diff 分析）
│   ├── tasks/          # Celery 异步任务
│   ├── core/           # 配置中心
│   └── utils/          # Redis / Tracing
├── tests/
│   ├── unit/           # 单元测试（13 条）
│   ├── integration/    # 集成测试（3 条）
│   └── eval/           # Ragas 评估（14 条测试集）
├── docs/               # 项目文档
├── config/             # 风险规则配置
├── static/             # 前端 UI
├── Dockerfile          # 多阶段构建
├── docker-compose.yml  # 6 服务编排
└── pyproject.toml
```

## 文档

- [API 文档](docs/api.md)
- [部署文档](docs/deploy.md)
- [项目章程](docs/project-charter.md)
- [进度记录](docs/PROGRESS.md)
