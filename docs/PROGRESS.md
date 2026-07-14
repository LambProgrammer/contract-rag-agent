# 项目进度记录

> 最后更新：2026-07-14

---

## 当前进行中

- **里程碑**：5. 工程化 + 评估 — **全部完成** ✅
- **状态**：已完成
- **下一步**：项目交付。后续优化方向见下文「生产环境部署规划（V2.0+）」

> 本地开发启动命令：
> ```bash
> # Docker Desktop: postgres + redis + qdrant 绿灯（app/worker 停掉）
> uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000       # 终端 1
> uv run celery -A src.tasks.celery_app worker --loglevel=info --pool=solo  # 终端 2
> # 浏览器: http://localhost:8000
> ```

---

## 已完成

- [x] **里程碑 0**：项目初始化 — 环境配置、Docker Compose 编排、包结构 `__init__.py`、CI lint 流水线全部就绪。基础设施三服务（PostgreSQL / Redis / Qdrant）容器化验证通过。
- [x] **里程碑 1**：上传 + 解析 — 上传→解析→分块→向量化→索引入库全链路打通，5 步骤共 15 个文件。验证：29 页/356KB PDF ↔ 30 个 Chunk ↔ 30 个 Qdrant Point，~101s。
- [x] **里程碑 2**：基础 RAG 问答 — 用户提问→检索→LLM 生成答案回路闭环，3 步骤共 7 个文件。验证：Swagger 手动走通上传→解析→索引→提问→答案带条款引用全闭环。
- [x] **里程碑 3**：核心优化 — 4 步骤全部完成：混合检索 + 重排序 + 查询改写+回退+多轮记忆 + SSE 流式。图结构：`rewrite → retrieve → rerank → check → generate/fallback`。
- [x] **里程碑 4**：特色功能 — 2 大步骤全部完成：
  - 步骤 A — 风险检测：`config/risk_rules.yaml`（13 条规则，覆盖民法典核心争议点）+ `rule_engine.py`（关键词粗筛）+ `llm_verifier.py`（LLM 二次确认+误报过滤）+ `router.py`（POST /api/v1/risk/detect，含 session_id + Redis 暂存）
  - 步骤 B — 跨合同比对：`diff_analyzer.py`（B0 全文相关性预检 + B1 条款对齐正则+语义回退 + B2 MD5 hash 预过滤相同内容 + B3 上下文扩充+LLM 逐句 diff 风险评估）+ `router.py`（POST /api/v1/compare，N=2，含 session_id + Redis 暂存 + 统计粒度）
  - QA 会话串联：`redis_client.py` 新增 `get_compare_context()`，QA 端点读取 risk/compare 快照串联上下文
- [x] **M5-E LangFuse 可观测性**：`src/utils/tracing.py` + 5 个 RAG 节点 `@observe()` + lifespan 资源管理（Redis/Qdrant 连接池化 + 启动/关闭生命周期）。最终采用 LangFuse Cloud（自托管 v2/v3 与 unstructured 存在 wrapt 传递性依赖冲突，详见下方 Bug 9）。
- [x] **M5-F Ragas 评估**：`tests/eval/test_ragas.py` + 14 条中文测试集（6 类场景，覆盖条款查询 / 阈值查询 / 指代消解 / 跨条款综合 / Bug 1&8 验证）。首次评估基线：
  - Faithfulness **0.67** / AnswerRelevancy **0.57** / ContextRecall **0.18** / ContextRelevance **0.71**
  - 四项指标分数上报 LangFuse，可结合 trace 链路定位问题。评估依赖传递性修复覆盖 7 个 Bug（9-15 号，ragas 0.4.3 兼容链）。
- [x] **M5-G 部署优化**：Docker 多阶段构建（镜像瘦身 ~15%）+ app/worker 容器 healthcheck + HF 模型缓存卷持久化 + .doc 格式 Unstructured 回退全链路测试通过（解析器=unstructured, 31 分块, 106s）。
- [x] **M5-H 文档完善**：`README.md`（项目简介+架构+快速开始）+ `docs/api.md`（6 个端点+请求/响应示例）+ `docs/deploy.md`（容器架构+部署步骤+模型下载策略+常见问题）。

---

## 已发现并修复的 Bug（开发全程积累）

> 记录开发过程中发现、分析、修复的 bug，供复盘参考。

| # | 发现阶段 | 问题 | 根因 | 修复方式 |
|------|:--:|------|------|------|
| 1 | M3 S3 | Cross-encoder 置信度门控永远不触发 | cross-encoder 得分是"相对最优排名"而非"绝对相关度"——只要候选池非空就必然有高分。这是一个数学上的错误用法 | 移除 cross-encoder 阈值判断；改为检索预检门控（dense<0.30 且 sparse<0.001）；`_check_confidence_node` 简化为仅空 docs fallback；最终裁决权交给 LLM 自身判断 |
| 2 | M2 | sentence-transformers 加载时 HF Hub 网络超时，进程 hang 20 分钟 | `SentenceTransformer()` 默认向 huggingface.co 发 HEAD 请求检查模型更新，国内网络不通 | `local_files_only=True` 优先本地缓存；`HF_HUB_OFFLINE=1` 双保险；本地无模型时抛出明确的错误提示而非无限重试 |
| 3 | M2 | Windows 宿主机加载 BGE 模型时段错误（segfault, exit 139） | torch 在 Windows CPU 模式下的 DLL 初始化顺序缺陷——sentence-transformers 直接 import torch 时 C++ 扩展未完整加载 | `src/core/windows_patch.py`：`from docling.document_converter import DocumentConverter` 预初始化 torch 运行时；`OMP_NUM_THREADS=1` 限制线程数 |
| 4 | M1 S5 | celery_app.py 与 process_contract.py 循环导入，ImportError | celery_app.py 导入 process_contract → process_contract.py 导入 celery_app → Python 解释器报 partially initialized module | `process_contract.py` 改用 `@shared_task` 替代 `@app.task`，不依赖 celery_app 模块级导入 |
| 5 | M2 | Docker Worker 收到 Windows FastAPI 传的文件路径后 FileNotFoundError | Windows Path → `str()` 输出反斜杠（`\`），Docker 容器内 Linux 内核不识别 | `upload.py` 中 `str(file_path)` → `file_path.as_posix()` |
| 6 | M2 | Docker 容器内 Unstructured 解析 PDF 报 ImportError: partition_pdf() not available | `pyproject.toml` 声明的 `unstructured>=0.23.1` 不含 PDF 解析引擎（pdfminer/pikepdf） | 改为 `unstructured[pdf]>=0.23.1` |
| 7 | M1 S5 | Qdrant client 版本与 server 不兼容警告：client 1.18 vs server 1.12 | `pyproject.toml` 中 `qdrant-client>=1.18.0` 与 `docker-compose.yml` 中 `v1.12.0` 差距超过 1 | 降级 client 为 `>=1.12.0,<1.14` |
| 8 | M3 S3 | `sparse_dead = not sparse_top1` 判空逻辑过弱，停用词偶发匹配导致误放行 | 合同文本中存在"什么""怎样"等通用词，jieba 分词后可能命中，sparse 路返回空列表的条件几乎不成立 | 加稀疏得分阈值 `sparse_top1[0].score < 0.001`，零分匹配等同于无匹配 |
| 9 | M5-E | LangFuse 自托管方案走不通，最终迁移 Cloud | 链条：SDK 4.x 通过 OTEL 发送 span → v2 server 不支持 OTEL（404）→ 降级 SDK v2 与 unstructured 的 wrapt 版本冲突 → 升 v3 server 需新增 ClickHouse 容器，过重。详见下一条 LangFuse API 适配 | 放弃自托管，改用 LangFuse Cloud（`LANGFUSE_HOST=https://cloud.langfuse.com`）；移除 docker-compose 中 langfuse-server 服务；`@observe()` 自动上报正常工作 |
| 10 | M5-E | LangFuse SDK 4.9.0 API 使用错误 → 评估分数上报失败 | `start_as_current_observation()` 返回 context manager 而非普通对象，`.score()` / `.end()` 方法不存在于该返回类型 | 改用 `with ... as:` + `langfuse.score_current_span()`；lifespan shutdown 中调 `langfuse.flush()` |
| 11 | M5-F | Ragas 0.4.3 依赖链兼容性问题（复合 Bug，含 5 个子问题） | (1) langchain-community 删除 vertexai 模块 → `import ragas` 硬失败 → `ragas_compat.py` stub；(2) 新版 collections 指标不继承旧版 `Metric` → `aevaluate()` 拒绝 → 切换 `from ragas.metrics` 旧版 API；(3) 旧版指标调 `agenerate_text()`，新版 `InstructorLLM` 无此方法 → 锁定 `LangchainLLMWrapper + ChatDeepSeek`；(4) 自写 `_BGERagasEmbedding` 逐次被调 `embed_query` / `embed_documents` → 补齐两个方法；(5) 旧版 Faithfulness NLI 阶段请求 `n=3` 生成，DeepSeek 不支持 → 仅影响 2/56 任务，`_safe_float` 降级 NaN | 最终锁定技术栈：旧版 ragas 指标 + `LangchainLLMWrapper(ChatDeepSeek)` + `_BGERagasEmbedding(BaseRagasEmbedding)` + `ragas_compat.py` stub；评估脚本 7 次迭代后通过，4 项指标全部有分数 |
| 12 | M5-G | 容器 PDF 解析失败：Docling 和 Unstructured 均报 `libGL.so.1: cannot open shared object file` | OpenCV（cv2）依赖 `libGL.so.1`，Dockerfile 中 `libgl1-mesa-dri` 提供了 DRI 驱动但未提供 libGL 本体 | 添加 `libgl1` 到 `apt-get install` 列表 |
| 13 | M5-G | Cross-Encoder 模型运行时下载失败，Dockerfile 预装了模型但仍然找不到 | `docker-compose.yml` 的 `hf_cache` 卷挂载到 `/root/.cache/huggingface`，空卷覆盖了镜像中预装的模型目录 | 移除 `hf_cache` 卷挂载（镜像预装模型后不再需要运行时卷）；Dockerfile 预装两个模型（BGE + Cross-Encoder）|

---

## M5 待办优化项（随开发过程积累）

| 优化项 | 说明 | 状态 |
|--------|------|:--:|
| **简易前端 UI** | 单页 HTML UI，覆盖全部 5 个 API 端点，支持 SSE 流式输出可视化 | ✅ |
| **risk/compare 的 QA 记忆验证** | 同一 session_id 下 risk→QA 追问和 compare→QA 追问均确认为上下文串联正确 | ✅ |
| **异步化改造** | `/risk/detect` 和 `/compare` 改为 Celery 异步任务 | ✅ |
| 文本清洗步骤 | `clean_text()` 函数 + 挂载解析流程 | ✅ |
| 文件上传大小校验 | `upload.py` 拒绝超限文件 | ✅ |
| `/health/ready` 连通性检查 | 跳过——当前项目无编排系统调用，不影响功能 | ⏭️ |
| 结构化日志 | `print()` → `logger`（celery_app.py + main.py） | ✅ |
| 参数调优（B0/对齐/阈值） | B0 阈值迁入 `.env`；`_find_by_clause_ref` 过滤无名标签；`_extract_clause_ref` 回退标签优化；对齐精度残余归入 V2.0 | ✅ |
| **UI 全功能大测试** | 5 轮全覆盖测试通过；测试中修复 `_generate_node` 上下文读取、Redis 快照全文化、分块器/比对器标签优化 | ✅ |
| `.doc` 格式 Unstructured 回退全链路测试 | 容器内测试通过：解析器=unstructured, 31 分块, 106s | ✅ |
| Docling 容器内模型下载策略 | 镜像预装 / HF 缓存挂载 / 国内镜像源 — 见 `docs/deploy.md` | ✅ |
| 单元测试 | `pytest tests/unit/` 覆盖核心模块（config、parsers、chunker、retrieval） | ✅ |
| 集成测试 | `pytest tests/integration/` 覆盖全链路（上传→解析→索引→QA） | ✅ |
| CI/CD 流水线 | `.github/workflows/ci.yml`（lint + test 并行）+ `cd.yml`（tag 触发 → docker build → push Docker Hub） | ✅ |
| LangFuse 可观测性 | `src/utils/tracing.py` + `@observe()` 追踪关键函数调用链 | ✅ |
| Ragas 评估 | `tests/eval/test_ragas.py`：对已索引合同跑 Faithfulness/Relevancy 等指标 | ✅ |
| Docker 镜像瘦身 | 多阶段构建、.dockerignore 优化、dev 依赖剔除 | ✅ |
| Worker 健康检查 | `docker-compose.yml` worker + app 容器补充 healthcheck | ✅ |
| 文档完善 | `README.md` + `docs/api.md` + `docs/deploy.md` | ✅ |

---

## 阻塞项

*暂无*

---

## 当前生效的技术决策

> ⚠️ 更新规则：如需修改某领域选型，请直接替换下表对应内容并更新日期。**禁止追加新行，禁止保留旧版本。**

| 领域 | 当前选型 | 确认日期 |
| :--- | :--- | :--- |
| PDF 解析 | Docling（主力）+ Unstructured（回退） | 2026-06-18 |
| 向量数据库 | Qdrant 1.12+（启用混合检索） | 2026-06-18 |
| LLM 提供商 | DeepSeek API (deepseek-v4-pro) | 2026-06-24 |
| Agent 框架 | LangGraph + LangChain | 2026-06-18 |
| API 框架 | FastAPI | 2026-06-18 |
| Embedding 模型 | BAAI/bge-small-zh-v1.5 | 2026-06-26 |
| 可观测性 | LangFuse Cloud（原计划自托管，因 wrapt 冲突放弃） | 2026-07-14 |
| 评估框架 | Ragas 0.4.3（旧版 API，因新版 collections 与 aevaluate 不兼容） | 2026-07-14 |

---

## 生产环境部署规划（V2.0+）

> 以下内容在当前开发阶段（M0-M5）中不涉及，标记为"若本项目部署至云服务器或上线生产环境时，需要补充的内容"。
> 当前项目定位为「学习与探索 RAG 技术的结对编程项目」，运行模式为本地 `uv run` + `docker compose up`。
> 若规划生产环境部署，以下能力需要逐一补齐。

### 数据库层

| 能力 | 说明 |
|------|------|
| PostgreSQL 数据库建表 | 当前数据全部存储在 Qdrant（条款/向量）和 Redis（任务状态/对话历史/比对快照），PostgreSQL 容器虽已在 docker-compose 中运行但未建表。生产环境需使用 PG 存储：① 合同元数据（上传时间、原始文件名、解析状态、页数/字数）；② 用户认证与多租户隔离；③ 操作审计日志；④ 对话历史与比对结果的持久化（突破 Redis TTL 限制） |
| 数据库连接池 | `main.py` lifespan 中创建 asyncpg 连接池，供 API 端点复用 |

### 功能补全

| 功能 | 说明 |
|------|------|
| 标准模板对比能力 | 预置不同合同类型的标准条款清单，逐条检查合同中是否覆盖应有条款，自动生成"合规缺口报告"。从"问答工具"升级为"合同合规体检工具" |
| 版本去重机制 | 同一合同 V1/V2 重复索引导致存储膨胀和比对语义混淆 |
| 条款层级树（条→款→项）对齐 | 当前线性分块无法保留层级结构，导致颗粒度对齐失准 |
| N≥3 多文档比对 | 三方协议比对时，"两两组合"还是"以第一份为基准"的模式需定义 |
| 批量/项目级风险扫描 | 当前仅支持单篇合同检测，扩展到一次扫描多份合同 |
| 风险规则库完善 | 从硬编码关键词扩展为可动态更新的规则管理后台；R003 等遗漏关键词补全 |

### 架构升级

| 项目 | 说明 |
|------|------|
| API 网关 / 认证 | 用户登录、权限分级（管理员/普通用户） |
| 持久化存储 | 对话历史、比对结果突破 Redis 1 小时 TTL，迁入 PG |
| 监控告警 | 接入 Prometheus + Grafana，对 API 延迟、错误率、Worker 积压设置告警 |
| 模型版本管理 | Embedding 模型升级时的向量迁移策略 |
| GPU 推理加速 | Docling PDF 解析当前 CPU 模式 29 页耗时 ~80s，GPU（CUDA）推理可将耗时降至 10-15s；Embedding 向量化批量推理亦可大幅提速。生产环境建议配置 GPU 节点或使用云 GPU 实例 |
| 意图路由（闲聊/合同分流） | 当前架构所有用户输入均走完整 RAG 管线（rewrite→retrieve→rerank→generate），非合同问题（打招呼、自我介绍等）也会被检索节点处理，导致空 docs 时 fallback 文本生硬。生产环境应在检索前加入轻量意图路由：识别闲聊后跳过检索直接 LLM 自由回应，合同问题再走完整 RAG 管线 |