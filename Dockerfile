# ============================================================
# Contract RAG Agent — Dockerfile
# 两种角色：API 服务（FastAPI） / Worker 服务（Celery）
# 通过 docker-compose command 区分启动方式
# ============================================================

FROM python:3.12-slim

# ---------- 系统依赖 ----------
# libgl1-mesa-dri    : Docling PDF 渲染所需
# libglib2.0-0       : Docling 运行时依赖
# poppler-utils      : Unstructured 回退解析 PDF
# tesseract-ocr      : Unstructured OCR 回退（中英文）
# libreoffice-writer : Unstructured 解析 .doc 格式（soffice 命令转 PDF 后再解析）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dri \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 安装 uv（快速 pip 安装器）----------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ---------- 先复制依赖文件（利用 Docker 层缓存）----------
COPY pyproject.toml uv.lock ./

# ---------- 安装 Python 依赖 ----------
# --frozen: 严格使用 uv.lock 锁定的版本
# --no-dev: 不安装 dev 依赖（pytest/ruff/pre-commit），缩减镜像体积
RUN uv sync --frozen --no-dev

# ---------- 复制项目源码 ----------
COPY . .

# ---------- 创建运行时目录 ----------
RUN mkdir -p /app/uploads /app/logs

EXPOSE 8000

# 默认命令（可被 docker-compose 覆盖）
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
