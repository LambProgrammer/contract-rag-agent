# 部署文档

## 容器架构

```
docker compose up -d
    │
    ├── contract-rag-api         :8000   FastAPI 应用
    ├── contract-rag-worker              Celery Worker（文档解析、风险检测、合同比对）
    ├── contract-rag-postgres    :5432   PostgreSQL 15
    ├── contract-rag-redis       :6379   Redis 7（Celery broker + 对话历史）
    ├── contract-rag-qdrant      :6333   Qdrant v1.12（向量检索）
    └── volumes:
        ├── postgres_data                数据库持久化
        ├── redis_data                   缓存持久化
        ├── qdrant_data                  向量索引持久化
        └── hf_cache                     HuggingFace 模型缓存（BGE ~400MB）
```

---

## 部署步骤

### 1. 配置环境变量

```bash
cp .env.example .env
```

必填项：

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `LANGFUSE_PUBLIC_KEY` | LangFuse Cloud Public Key（可选，不配则跳过追踪） |
| `LANGFUSE_SECRET_KEY` | LangFuse Cloud Secret Key（可选） |

### 2. 启动

```bash
docker compose up -d --build
```

首次构建约 10-20 分钟（取决于网络）。后续构建利用 Docker 层缓存，数分钟完成。

### 3. 验证

```bash
# 检查容器状态
docker compose ps

# 健康检查
curl http://localhost:8000/health
```

打开浏览器：http://localhost:8000

### 4. 停止

```bash
# 停止容器（保留数据）
docker compose stop

# 停止并删除容器（保留数据卷）
docker compose down

# 停止并删除一切（含数据）
docker compose down -v
```

---

## HuggingFace 模型下载策略

项目使用了两个 HuggingFace 模型：

| 模型 | 大小 | 用途 |
|------|------|------|
| BAAI/bge-small-zh-v1.5 | ~400MB | 稠密向量嵌入（Worker 解析时加载） |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | ~80MB | 检索结果重排序（App 首次 QA 时加载） |

### 策略 1：Dockerfile 预装（当前方案，通用）

构建阶段预下载模型，打进镜像。运行时零网络依赖，任何环境直接可用：

```dockerfile
# Dockerfile builder 阶段，uv sync 之后：
RUN uv run python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
"
```

国内构建时可通过 `--build-arg` 指定 HuggingFace 镜像源：

```bash
docker compose build --build-arg HF_MIRROR=https://hf-mirror.com
```

代价：镜像体积增加 ~480MB。**这是当前项目采用的方案。**

### 策略 2：HF 缓存卷挂载（开发备选）

`docker-compose.yml` 中配置了 `hf_cache` 卷，挂载到容器的 `/root/.cache/huggingface`。首次下载后持久化，重建容器不需重新下载。适用于策略 1 未启用时的开发场景。

### 策略 3：国内镜像源（本地开发备选）

使用 `uv run` 本地开发时，如 HuggingFace 不可达，可设环境变量：

```env
HF_ENDPOINT=https://hf-mirror.com
```

> 国内镜像源的有效性随时间变化。

---

## 开发模式

不构建 Docker，本地直接跑：

```bash
# 1. 只启动基础设施
docker compose up -d postgres redis qdrant

# 2. 终端 1 — FastAPI（热重载）
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 3. 终端 2 — Celery Worker
uv run celery -A src.tasks.celery_app worker --loglevel=info --pool=solo
```

---

## CI/CD

- **CI**：push 到 main 分支自动触发 ruff lint + 单元测试
- **CD**：push `v*` tag 自动构建 Docker 镜像并推送到 Docker Hub

GitHub Secrets 需配置：`DOCKER_USERNAME`、`DOCKER_PASSWORD`

---

## 常见问题

| 问题 | 解决 |
|------|------|
| 容器启动后 8000 端口访问不了 | 确认宿主机 8000 端口未被占用；等待 Worker 完全启动（约 30s） |
| Worker 日志显示 HuggingFace 下载超时 | 模型已预装于镜像中，通常无需运行时下载。如确需下载，参考上方策略 3 |
| 上传合同后长时间处于 processing | 查看 Worker 容器日志；首次需下载 embedding 模型约 1 分钟 |
| `docker compose build` 卡在拉取基础镜像 | 国内网络问题，重试多次或配置镜像加速器 |
