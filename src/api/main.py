"""
FastAPI 应用入口
================
在 RAG 系统中的角色：
    这是整个系统的"门面"——所有用户请求（上传合同、提问、查询任务状态）
    都通过这个 FastAPI 实例接入。它不直接执行文档解析（那是 Worker 的活），
    而是负责接收请求、参数校验、提交异步任务、返回结果。

生命周期（lifespan）：
    启动时 → 初始化 LangFuse 可观测性 → 创建共享 Redis/Qdrant 连接 → 启动服务
    关闭时 → 刷新 LangFuse 缓冲区 → 关闭 Redis 连接池 → 关闭 Qdrant 连接

路由挂载策略：
    - /api/v1/*    业务路由（upload、qa、compare 等）
    - /health       健康检查（容器编排用）
    - /docs         Swagger UI（开发调试用）

用法（容器内启动）：
    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis as redis_lib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from src.core.config import settings  # noqa: E402

# 项目日志输出到 stdout（uvicorn 只给自己的 logger 加了 handler）
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
_log_pkg = logging.getLogger("src")
_log_pkg.addHandler(_log_handler)
_log_pkg.setLevel(getattr(logging, settings.app_log_level, logging.INFO))
_log_pkg.propagate = False  # 不重复输出到 root handler

logger = logging.getLogger(__name__)


# ============================================================
# 生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 生命周期上下文管理器。

    启动阶段（yield 之前）：
        - 打印服务配置信息
        - 初始化 LangFuse 可观测性（失败不影响主业务）
        - 创建共享 Redis 连接（挂载到 app.state.redis）
        - 创建共享 Qdrant 客户端（挂载到 app.state.qdrant）

    关闭阶段（yield 之后）：
        - 刷新 LangFuse 缓冲区
        - 关闭 Redis 连接池
        - 关闭 Qdrant 客户端
    """
    # ============================================================
    # 启动阶段
    # ============================================================
    logger.info(f"[APP] ========== {settings.app_name} 启动 ==========")
    logger.info(f"[APP] Debug 模式: {settings.app_debug}")
    logger.info(f"[APP] PostgreSQL: {settings.postgres_host}:{settings.postgres_port}")
    logger.info(f"[APP] Redis: {settings.redis_host}:{settings.redis_port}")
    logger.info(f"[APP] Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"[APP] Embedding 模型: {settings.embedding_model}")
    logger.info(f"[APP] DeepSeek 模型: {settings.deepseek_model}")
    logger.info(f"[APP] 上传目录: {settings.upload_dir}")
    logger.info("[APP] ==============================================")

    # ---- 初始化 LangFuse 可观测性 ----
    from src.utils.tracing import init_langfuse, shutdown_langfuse

    langfuse_client = init_langfuse()
    app.state.langfuse = langfuse_client  # 可能为 None（降级）

    # ---- 创建共享 Redis 连接 ----
    redis_client = redis_lib.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )
    app.state.redis = redis_client
    # 注入到 redis_client 工具模块，使 API 路由不需要每次都新建连接
    from src.utils.redis_client import set_shared_redis

    set_shared_redis(redis_client)
    logger.info("[APP] Redis 共享连接已就绪")

    # ---- 创建共享 Qdrant 客户端 ----
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=False,
    )
    app.state.qdrant = qdrant_client
    from src.indexing.vector_client import set_shared_qdrant_client

    set_shared_qdrant_client(qdrant_client)
    logger.info("[APP] Qdrant 共享客户端已就绪")

    # 生产环境顺延：asyncpg 数据库连接池（见 PROGRESS.md「生产环境部署规划」）

    yield  # ← 服务在这里运行

    # ============================================================
    # 关闭阶段
    # ============================================================
    logger.info(f"[APP] ========== {settings.app_name} 关闭 ==========")

    # ---- 刷新 LangFuse 缓冲区 ----
    shutdown_langfuse()

    # ---- 关闭 Redis 连接池 ----
    if hasattr(app.state, "redis") and app.state.redis:
        try:
            app.state.redis.close()
            logger.info("[APP] Redis 连接池已关闭")
        except Exception as e:
            logger.warning(f"[APP] Redis 关闭异常: {e}")

    # ---- 关闭 Qdrant 客户端 ----
    if hasattr(app.state, "qdrant") and app.state.qdrant:
        try:
            app.state.qdrant.close()
            logger.info("[APP] Qdrant 客户端已关闭")
        except Exception as e:
            logger.warning(f"[APP] Qdrant 关闭异常: {e}")


# ============================================================
# FastAPI 应用实例
# ============================================================
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="智能合同审查 RAG Agent — 上传、解析、检索、风险检测",
    docs_url="/docs",  # Swagger UI 地址
    redoc_url="/redoc",  # ReDoc 地址
    lifespan=lifespan,  # 生命周期管理
)


# ============================================================
# 基础健康检查（不依赖任何外部服务）
# ============================================================
@app.get("/health", tags=["系统"])
async def health_check():
    """
    健康检查端点 — Docker 容器编排使用。

    返回 HTTP 200 表示 FastAPI 进程正常。
    注意：这里不检查 DB/Redis/Qdrant 连通性，因为这只是一个存活探针（liveness probe）。
    如果需要就绪探针（readiness probe），访问 /health/ready。
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": "0.1.0",
    }


@app.get("/health/ready", tags=["系统"])
async def readiness_check():
    """
    就绪检查端点 — 当前为**占位实现**，尚未检查外部依赖连通性。

    现状说明：
        三个依赖项固定返回 "pending"，不做任何实际探测。这是有意为之：
        当前项目以 `uv run` + `docker compose` 本地运行，没有 K8s 之类的
        编排系统会消费就绪探针，因此按 docs/PROGRESS.md「M5 待办优化项」
        的结论跳过（该项标记为 ⏭️）。
        容器健康检查统一使用 /health（存活探针，见 docker-compose.yml）。

    与 /health 的区别（如后续补齐探测）：
        - /health 只确认进程在跑（存活探针）
        - /health/ready 应确认可以接流量（就绪探针：DB、Redis、Qdrant 都通）
    """
    # 有意未实现依赖探测，原因见上方 docstring（非待办事项）
    return {
        "status": "ready",
        "checks": {
            "postgres": "pending",  # 未实现（有意跳过）
            "redis": "pending",  # 未实现（有意跳过）
            "qdrant": "pending",  # 未实现（有意跳过）
        },
    }


# ============================================================
# 路由注册（逐步挂载，每个里程碑解锁一部分）
# ============================================================
from src.api.upload import router as upload_router  # noqa: E402

app.include_router(upload_router)

from src.api.qa import router as qa_router  # noqa: E402

app.include_router(qa_router)

from src.risk.router import router as risk_router  # noqa: E402

app.include_router(risk_router)

from src.comparator.router import router as compare_router  # noqa: E402

app.include_router(compare_router)


# ============================================================
# 前端 UI（开发者自测用）
# ============================================================
import os as _os  # noqa: E402

_static_dir = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))), "static"
)
if _os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
