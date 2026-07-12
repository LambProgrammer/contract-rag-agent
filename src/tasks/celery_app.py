"""
Celery 应用定义
===============
在 RAG 系统中的角色：
    这是系统的"异步引擎"——负责在后台执行所有耗时操作。
    FastAPI 收到用户上传请求后，只做两件事：
        1. 保存文件到磁盘
        2. 向 Celery 提交一个异步任务
    然后立即返回 task_id 给用户，不阻塞 HTTP 连接。

    真正的重活（PDF 解析、文本分块、Embedding 向量化、写入 Qdrant）
    全部由 Celery Worker 在后台完成，用户可以通过 task_id 轮询进度。

为什么 RAG 需要异步任务队列：
    - PDF 解析（Docling）可能需要 10-30 秒，期间不能阻塞 HTTP 请求
    - Embedding 向量化（sentence-transformers）是 CPU 密集型操作
    - 多用户同时上传时，需要队列公平调度
    - 失败重试：PDF 解析偶尔因文件格式报错，Celery 支持自动重试

架构（与 docker-compose.yml 对应）：
    FastAPI (app) ──提交任务──→ Redis (broker) ──取任务──→ Celery Worker
                                                              │
                                    结果存回 Redis (backend) ←┘
                                                              │
    FastAPI (app) ──查询结果──→ Redis (backend) ──返回结果──→ 用户

用法（容器内启动 Worker）：
    uv run celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4
"""

import logging

from celery import Celery

logger = logging.getLogger(__name__)

from src.core.config import settings  # noqa: E402

# ============================================================
# Celery 应用实例
# ============================================================
# broker: 消息代理 — 任务队列存在这里（FastAPI 往这写，Worker 从这读）
# backend: 结果存储 — 任务执行结果存这里（FastAPI 来这查）
app = Celery(
    "contract_rag",
    broker=settings.celery_broker_url,  # redis://redis:6379/0
    backend=settings.celery_result_backend,  # redis://redis:6379/0（同一个 Redis）
)

# ============================================================
# Celery 全局配置
# ============================================================
app.conf.update(
    # ----- 序列化 -----
    # JSON 格式保证跨语言可读，调试时可以直接在 Redis 里查看任务内容
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # ----- 时区 -----
    timezone="Asia/Shanghai",
    enable_utc=True,
    # ----- 任务时效 -----
    # 超过 soft_time_limit 触发 SoftTimeLimitExceeded（任务可捕获做清理）
    # 超过 time_limit 强制 SIGKILL（无法捕获）
    task_soft_time_limit=settings.celery_task_soft_time_limit,  # 默认 240s
    task_time_limit=settings.celery_task_time_limit,  # 默认 300s
    # ----- 结果过期 -----
    # 任务结果在 Redis 中保留 1 小时，过期自动删除（避免 Redis 内存撑爆）
    result_expires=3600,
    # ----- 确认机制 -----
    # 任务完成后才确认（不会丢任务），但可能重复执行
    task_acks_late=True,
    # 每个 Worker 最多同时执行 4 个任务
    worker_prefetch_multiplier=1,
    # ----- 重试策略 -----
    # 默认最多重试 3 次，指数退避（第 1 次等 10s，第 2 次等 20s，第 3 次等 40s）
    task_default_retry_delay=60,  # 重试间隔基础值（秒）
    task_max_retries=3,
)

# ============================================================
# 显式注册任务（Import 即注册）
# ============================================================
# 必须显式导入任务模块，确保 Celery Worker 启动时任务已注册。
# 不使用 autodiscover_tasks 的原因：
#   - process_contract 在 autodiscover 扫描时可能尚未创建，
#     导致 Worker 启动时报"未找到任务"警告
#   - 显式导入更可控，新增任务时只需加一行 import
from src.tasks.compare_task import process_comparison  # noqa: E402, F401
from src.tasks.process_contract import process_contract  # noqa: E402, F401
from src.tasks.risk_task import process_risk_detection  # noqa: E402, F401

# 启动时打印关键配置（确认容器内环境变量正确）
logger.info(f"[Celery] Broker: {settings.celery_broker_url}")
logger.info(f"[Celery] Backend: {settings.celery_result_backend}")
logger.info(f"[Celery] 并发数: {settings.celery_worker_concurrency}")
logger.info(f"[Celery] 任务超时: {settings.celery_task_time_limit}s")
