"""
LangFuse 可观测性模块
====================
在 RAG 系统中的角色：
    为 RAG 管线的关键节点（检索、重排序、生成）提供分布式追踪能力，
    让每次用户请求的完整调用链可追溯、可度量、可优化。

降级策略：
    LangFuse 不可用时静默降级，所有 @observe() 装饰器自动跳过，
    不影响主业务流程。这是"可观测性是增强，不是依赖"原则的落地。

用法：
    from src.utils.tracing import init_langfuse, shutdown_langfuse

    # 在 FastAPI lifespan startup 中调用 init_langfuse()
    # 在 FastAPI lifespan shutdown 中调用 shutdown_langfuse()
    # 其他模块直接 from langfuse import observe 装饰函数即可
"""

import logging

from langfuse import Langfuse

from src.core.config import settings

logger = logging.getLogger(__name__)

# 全局单例 — 模块级私有，通过 get_langfuse() 访问
_langfuse_client: Langfuse | None = None


def init_langfuse() -> Langfuse | None:
    """
    初始化 LangFuse 客户端并设置全局 OpenTelemetry trace provider。

    调用时机：FastAPI lifespan startup 阶段。
    调用次数：整个应用生命周期内仅调用一次。

    降级行为：
        - 未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY → 跳过，打印提示
        - 网络不通或 LangFuse 服务未启动 → 跳过，打印 warning
        - 任何异常都不向上传播，确保主业务不中断

    返回：
        初始化成功的 Langfuse 实例，或降级时返回 None
    """
    global _langfuse_client

    # ---- 检查是否已配置密钥 ----
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info(
            "[LangFuse] 未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，"
            "可观测性功能不可用。如需启用：\n"
            "  1. 在 .env 中配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY\n"
            "  2. 启动 LangFuse 容器：docker compose --profile langfuse up -d\n"
            "  3. 访问 http://localhost:3000 创建 API Keys"
        )
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info(
            f"[LangFuse] 已连接: {settings.langfuse_host} "
            f"(项目: {settings.langfuse_public_key[:12]}...)"
        )
        return _langfuse_client

    except Exception as e:
        logger.warning(
            f"[LangFuse] 初始化失败，追踪功能不可用。主业务流程不受影响。错误详情: {e}"
        )
        _langfuse_client = None
        return None


def get_langfuse() -> Langfuse | None:
    """获取全局 LangFuse 客户端（供需要手动创建 span 的模块使用）"""
    return _langfuse_client


def shutdown_langfuse() -> None:
    """
    优雅关闭 LangFuse — 将缓冲区中剩余的 trace 数据发送到服务端。

    调用时机：FastAPI lifespan shutdown 阶段。
    如果不调用 flush()，进程退出时缓冲区中的 trace 数据会丢失。
    """
    global _langfuse_client
    if _langfuse_client:
        try:
            _langfuse_client.flush()
            logger.info("[LangFuse] 缓冲区已刷新，追踪数据已发送")
        except Exception as e:
            logger.warning(f"[LangFuse] 缓冲区刷新失败: {e}")
        finally:
            _langfuse_client = None
