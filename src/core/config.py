"""
配置中心模块
============
在 RAG 系统中的角色：
    集中管理所有环境变量和应用参数，是整个系统的"配置大脑"。
    所有其他模块（FastAPI、Celery、Embedding 加载器、Qdrant 客户端等）
    都通过这一个入口获取配置，避免 os.getenv() 散落在各处。

为什么需要它：
    - 类型安全：端口号是 int 不是 str，dataclass 保证类型清晰
    - 默认值：embedding 模型名等有合理默认值，减少 .env 中的必填项
    - 拼接 URL：Celery broker 等需要从 host/port/db 拼接，不重复计算
    - 测试便利：测试时可以实例化 Settings() 并覆盖个别字段

用法：
    from src.core.config import settings
    print(settings.postgres_host)  # "localhost" 或 "postgres"
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# 加载 .env 文件（优先级：已有的环境变量 > .env）
load_dotenv()

# ---- Windows 平台兼容补丁 ----
# 两个步骤缺一不可（已验证）：
# 1. OMP_NUM_THREADS=1 — 限制 torch 的 OpenMP 线程数
# 2. import docling — 预初始化 torch C++ 运行时（DLL 加载顺序）
os.environ.setdefault("OMP_NUM_THREADS", "1")
from src.core.windows_patch import apply_windows_patches  # noqa: E402

apply_windows_patches()


@dataclass
class Settings:
    """应用配置聚合类 — 从环境变量读取，提供类型安全的访问"""

    # ============================================================
    # LLM 提供商（DeepSeek，通过 langchain-deepseek 调用）
    # ============================================================
    # langchain-deepseek 内部封装了 API endpoint，所以不需要配置 base_url
    # 只需要 API Key 和模型名即可初始化 ChatDeepSeek(model=..., api_key=...)
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_model: str = field(
        # deepseek-v4-pro 可通过 langchain-deepseek 直接调用
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    )

    # ============================================================
    # PostgreSQL 数据库
    # ============================================================
    postgres_host: str = field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost")
    )
    postgres_port: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432"))
    )
    postgres_user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "postgres")
    )
    postgres_password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres")
    )
    postgres_db: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "contract_rag")
    )

    # ============================================================
    # Redis（Celery broker + 结果缓存 + 语义缓存）
    # ============================================================
    redis_host: str = field(
        default_factory=lambda: os.getenv("REDIS_HOST", "localhost")
    )
    redis_port: int = field(
        default_factory=lambda: int(os.getenv("REDIS_PORT", "6379"))
    )
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))

    # ============================================================
    # Qdrant 向量数据库
    # ============================================================
    qdrant_host: str = field(
        default_factory=lambda: os.getenv("QDRANT_HOST", "localhost")
    )
    qdrant_port: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    qdrant_collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "contract_chunks")
    )

    # ============================================================
    # LangFuse 可观测性
    # ============================================================
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    )
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )

    # ============================================================
    # Embedding 模型
    # ============================================================
    # BAAI/bge-small-zh-v1.5：中文小模型（102M 参数），平衡速度与效果
    # 向量维度：512
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    # 跨编码器用于 M3 阶段的重排序（cross-encoder 比 bi-encoder 更准确但更慢）
    cross_encoder_model: str = field(
        default_factory=lambda: os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # ============================================================
    # 应用基础配置
    # ============================================================
    app_name: str = field(
        default_factory=lambda: os.getenv("APP_NAME", "Contract RAG Agent")
    )
    app_debug: bool = field(
        default_factory=lambda: os.getenv("APP_DEBUG", "false").lower() == "true"
    )
    app_log_level: str = field(
        default_factory=lambda: os.getenv("APP_LOG_LEVEL", "INFO")
    )
    api_v1_prefix: str = field(
        default_factory=lambda: os.getenv("API_V1_PREFIX", "/api/v1")
    )
    max_upload_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))
    )
    upload_dir: str = field(
        default_factory=lambda: os.getenv("UPLOAD_DIR", "./uploads")
    )

    # ============================================================
    # 检索参数（M2 及以后使用）
    # ============================================================
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "20"))
    )
    rerank_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_K", "5"))
    )
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    compare_relevance_threshold: float = field(
        default_factory=lambda: float(os.getenv("COMPARE_RELEVANCE_THRESHOLD", "0.55"))
    )

    # ============================================================
    # 分块参数
    # ============================================================
    min_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("MIN_CHUNK_SIZE", "100"))
    )
    max_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_CHUNK_SIZE", "1500"))
    )

    # ============================================================
    # Celery 任务配置
    # ============================================================
    celery_task_time_limit: int = field(
        default_factory=lambda: int(os.getenv("CELERY_TASK_TIME_LIMIT", "300"))
    )
    celery_task_soft_time_limit: int = field(
        default_factory=lambda: int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "240"))
    )
    celery_worker_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CELERY_WORKER_CONCURRENCY", "4"))
    )

    # ============================================================
    # 拼接字段（从基础字段计算得出，不需要在 .env 中配置）
    # ============================================================

    @property
    def database_url(self) -> str:
        """异步数据库连接 URL — FastAPI 连接 PostgreSQL 使用"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def celery_broker_url(self) -> str:
        """Celery 消息代理 URL — Redis 作为 broker"""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def celery_result_backend(self) -> str:
        """Celery 结果存储 URL — Redis 作为 result backend"""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def qdrant_url(self) -> str:
        """Qdrant REST API 地址"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


# 全局单例 — 整个应用共享一个配置实例
# 导入方式：from src.core.config import settings
settings = Settings()
