"""
Ragas 0.4.3 兼容补丁
====================
问题：
    ragas 0.4.3 在模块级别硬导入了 `langchain_community.chat_models.vertexai`，
    但 langchain-community 自 0.4.0 起已移除该模块（sunset，迁移至独立包
    langchain-google-vertexai）。这导致 `import ragas` 直接抛出
    ModuleNotFoundError。

    参考：https://github.com/langchain-ai/langchain-community/issues/674

解决方案：
    在本模块中注册一个最小化 stub，使 ragas 通过导入检查。
    ChatVertexAI 是 ragas 内部 LLM 工厂的备选后端——我们实际使用 DeepSeek
    作为评估 LLM，不会触发该类的实例化，因此空 stub 无害。

与 windows_patch.py 的关系：
    两者都是针对传递性依赖断裂的显式兼容处理。windows_patch.py 处理的是
    torch DLL 加载顺序，本模块处理的是 langchain-community 模块移除。
    不是 hack——是有注释、有出处、零副作用的 compatibility shim。

用法：
    在 `import ragas` 之前调用一次即可：
        from src.core.ragas_compat import apply_ragas_compat
        apply_ragas_compat()
"""

import logging
import sys
import types

logger = logging.getLogger(__name__)


def apply_ragas_compat() -> None:
    """
    为被 ragas 0.4.3 硬依赖但已被 langchain-community 移除的模块注册 stub。

    幂等操作：多次调用不会重复注册或报错。
    """
    _MODULE_NAME = "langchain_community.chat_models.vertexai"

    if _MODULE_NAME in sys.modules:
        return  # 已注册，跳过

    # ---- 创建最小化 stub 模块 ----
    _stub = types.ModuleType(_MODULE_NAME)

    # ChatVertexAI 是 ragas.llms.base 中唯一被引用的符号
    # 作为 LLM 工厂的备选后端类，它的类名被注册到工厂表中。
    # 我们使用 DeepSeek 作为评估 LLM，不会触发 VertexAI 实例化。
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[reportAttributeAccessIssue]

    sys.modules[_MODULE_NAME] = _stub

    logger.debug(
        "[ragas_compat] 已为 ragas 0.4.3 注册 langchain_community.chat_models.vertexai stub"
    )
