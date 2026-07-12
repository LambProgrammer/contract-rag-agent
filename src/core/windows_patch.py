"""
Windows 平台兼容补丁
====================
在 RAG 系统中的角色：
    解决 Windows 宿主机环境下 torch 运行时的 DLL 初始化顺序问题。
    不影响 Docker/Linux 环境，不影响任何业务逻辑。

背景：
    Windows 上 sentence-transformers 加载 BGE 模型时段错误（segfault），
    需要 docling 先初始化 torch 的完整运行时（包括 RapidOCR 的图像处理后端）。
    这个 import 只触发模块加载和 DLL 解析，不创建任何 AI 模型实例，不占内存。

为什么单独抽出来：
    - 避免 embedder.py 直接依赖 docling（检索层不应依赖文档解析层）
    - 如果将来 torch 修复了这个问题，只需删除这一个文件
    - 可以在此文件中集中管理所有平台兼容补丁

用法：
    在 src/core/config.py 顶部调用 apply_windows_patches()。
    该函数在非 Windows 平台上为空操作。
"""

import logging
import sys

logger = logging.getLogger(__name__)


def apply_windows_patches() -> None:
    """
    应用 Windows 平台的兼容性补丁。

    当前补丁：
        - 预加载 docling 以初始化 torch C++ 运行时，
          避免 sentence-transformers 加载 BGE 模型时段错误。

    非 Windows 平台：直接返回，不做任何操作。
    """
    if sys.platform != "win32":
        return

    try:
        # 必须导入 docling.document_converter 而非仅 import docling。
        # docling 包的 __init__.py 是轻量级的，不会触发 torch 相关子模块的加载。
        # DocumentConverter 的导入会级联加载 RapidOCR → torch 完整后端，
        # 这才是 sentence-transformers 需要的前置初始化。
        from docling.document_converter import DocumentConverter  # noqa: F401
    except ImportError:
        # 如果 docling 未安装（极罕见），不做处理。
        # 后续 sentence-transformers 加载失败时会报明确的错误。
        pass
