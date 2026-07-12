"""
FastAPI 应用入口
================
在 RAG 系统中的角色：
    这是整个系统的"门面"——所有用户请求（上传合同、提问、查询任务状态）
    都通过这个 FastAPI 实例接入。它不直接执行文档解析（那是 Worker 的活），
    而是负责接收请求、参数校验、提交异步任务、返回结果。

生命周期（lifespan）：
    启动时 → 验证数据库可连接、确保 Qdrant 可达
    运行中 → 接收 HTTP 请求，路由到对应的处理函数
    关闭时 → 断开数据库连接池，清理临时文件

路由挂载策略（为 M2/M3 预留扩展点）：
    - /api/v1/*    业务路由（upload、qa、compare 等）
    - /health       健康检查（容器编排用）
    - /docs         Swagger UI（开发调试用）

用法（容器内启动）：
    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

from src.core.config import settings  # noqa: E402


# ============================================================
# 生命周期管理
# ============================================================
# lifespan 是 FastAPI 推荐的启动/关闭钩子，替代已废弃的 on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 生命周期上下文管理器。

    启动阶段（yield 之前）：
        - 验证配置是否加载正确
        - 打印服务信息（用于确认容器内环境变量是否正确覆盖）

    关闭阶段（yield 之后）：
        - 清理数据库连接池（M1 之后实现）
        - 清理临时文件
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

    # 生产环境顺延：asyncpg 数据库连接池（见 PROGRESS.md「生产环境部署规划」）
    # TODO M5: 在这里初始化 LangFuse tracer
    yield  # ← 服务在这里运行
    # ============================================================
    # 关闭阶段
    # ============================================================
    logger.info(f"[APP] ========== {settings.app_name} 关闭 ==========")
    # TODO M1: 在这里关闭数据库连接池


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
    就绪检查端点 — 检查所有外部依赖是否可达。

    与 /health 的区别：
        - /health 只确认进程在跑（存活探针）
        - /health/ready 确认可以接流量（就绪探针：DB、Redis、Qdrant 都通）
    """
    # TODO M1: 添加 DB/Redis/Qdrant 连通性检查
    return {
        "status": "ready",
        "checks": {
            "postgres": "pending",  # M1 实现
            "redis": "pending",  # M1 实现
            "qdrant": "pending",  # M1 实现
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
