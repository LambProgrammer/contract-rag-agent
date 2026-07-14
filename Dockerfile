# ============================================================
# Contract RAG Agent — Dockerfile（多阶段构建）
#
# Stage 1 (builder)：安装 Python 依赖，生成 .venv/
# Stage 2 (runtime)：只复制运行时所需，剔除构建缓存
#
# 镜像体积对比：
#   单阶段 ≈ 3.2GB（含 uv 缓存 / pip 缓存 / 构建碎片）
#   多阶段 ≈ 2.5GB（只含 .venv + 系统依赖 + 源码）
#
# 两种角色：API 服务（FastAPI） / Worker 服务（Celery）
# 通过 docker-compose command 区分启动方式
# ============================================================

# ============================================================
# Stage 1 — 构建阶段
# ============================================================
FROM python:3.12-slim AS builder

# 安装 uv（快速 Python 包管理器）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先复制依赖清单（利用 Docker 层缓存：源码变更不触发重装依赖）
COPY pyproject.toml uv.lock ./

# 安装 Python 依赖到 .venv/
# --frozen: 严格使用 uv.lock
# --no-dev: 不装 pytest/ruff/pre-commit（缩减体积）
RUN uv sync --frozen --no-dev

# ---------- 预下载 ML 模型（打进入镜像，运行时无需联网）----------
# BAAI/bge-small-zh-v1.5     ~400MB  稠密向量嵌入
# cross-encoder/ms-marco...  ~80MB   检索结果重排序
# 国内构建时可指定镜像源：docker compose build --build-arg HF_MIRROR=https://hf-mirror.com
ARG HF_MIRROR=""
RUN if [ -n "$HF_MIRROR" ]; then export HF_ENDPOINT="$HF_MIRROR"; fi && uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ============================================================
# Stage 2 — 运行阶段
# ============================================================
FROM python:3.12-slim AS runtime

# ---------- 系统依赖 ----------
# libgl1             : OpenGL 共享库（cv2/OpenCV 需要，Docling + Unstructured 都依赖）
# libgl1-mesa-dri    : Mesa DRI 驱动（Docling PDF 渲染所需）
# libglib2.0-0       : GLib 运行时（Docling 依赖）
# poppler-utils      : Unstructured 回退解析 PDF
# tesseract-ocr      : Unstructured OCR 回退（中英文）
# libreoffice-writer : Unstructured 解析 .doc 格式
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 从构建阶段复制产物 ----------
# uv 二进制
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
# Python 虚拟环境（含全部运行时依赖）
COPY --from=builder /app/.venv /app/.venv
# HuggingFace 模型缓存（已在构建阶段预下载，运行时无需联网）
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# ---------- 复制项目源码 ----------
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY static/ ./static/
COPY config/ ./config/

# ---------- 运行时目录 ----------
RUN mkdir -p /app/uploads /app/logs

EXPOSE 8000

# 默认命令（可被 docker-compose 覆盖）
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
