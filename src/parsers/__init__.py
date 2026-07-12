"""
文档解析器模块（Parser）
=======================
在 RAG 系统中的角色：
    将用户上传的原始文件（PDF / Word）转换为结构化文本。
    这是 RAG 数据管道的第一道工序——输入是二进制文件，输出是 Markdown 文本。
    解析质量直接决定后续分块和检索的准确性。

解析器选择策略（工厂模式）：
    Docling（主力）→ 成功？→ 返回结果
                  → 失败？→ Unstructured（回退）→ 返回结果
    无论用哪个解析器，输出都是统一的 DocumentParseResult 格式，
    步骤 4 的分块器不需要知道底层用了哪个解析器。

为什么用 Markdown 而非纯文本：
    Docling 能识别文档结构（标题、表格、列表）并输出 Markdown。
    合同文本中，条款标题（"第一条"、"1.1"）会被转为 ## 或 ### 标题，
    后续分块器可以利用这些 Markdown 标记做更精确的结构化切割。

用法：
    from src.parsers import parse_document
    result = parse_document(Path("uploads/abc/合同.pdf"), doc_id="abc")
    print(result.text)         # Markdown 结构化文本
    print(result.page_count)   # 页数
    print(result.parser_used)  # "docling" 或 "unstructured"
"""

import logging
import time  # noqa: F401
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentParseResult:
    """
    文档解析的统一输出格式。

    无论 Docling 还是 Unstructured，都返回这个结构。
    步骤 4（分块器）和步骤 5（Celery 任务）只依赖这个接口，
    不关心底层解析器的具体实现。
    """

    doc_id: str
    """关联的文档 ID，与上传时的 doc_id 对应"""

    text: str
    """Markdown 格式的结构化文本，保留标题/表格/列表结构"""

    page_count: int
    """文档总页数，用于日志和元数据展示"""

    parse_time_ms: float
    """解析耗时（毫秒），用于性能监控"""

    parser_used: str
    """实际使用的解析器名称："docling" 或 "unstructured" """


def parse_document(file_path: Path, doc_id: str) -> DocumentParseResult:
    """
    自动选择解析器解析文档。

    策略：
        1. 先尝试 Docling（对中文合同 PDF 效果更好）
        2. 如果 Docling 抛出异常，自动回退到 Unstructured
        3. 如果两者都失败，抛出 RuntimeError

    参数：
        file_path: 上传文件的本地路径（例如 uploads/abc123/合同.pdf）
        doc_id: 文档唯一 ID（用于日志追踪和结果关联）

    返回：
        DocumentParseResult，包含 Markdown 文本和元数据
    """
    logger.info(f"[解析] 开始解析文档 {doc_id}: {file_path.name}")

    # ===================================================================
    # 策略 1：Docling 主力解析
    # ===================================================================
    try:
        from src.parsers.docling_parser import DoclingParser

        parser = DoclingParser()
        result = parser.parse(file_path, doc_id)
        from src.parsers.text_cleaner import clean_text

        result.text = clean_text(result.text)
        logger.info(
            f"[解析] Docling 成功: {doc_id}, "
            f"耗时 {result.parse_time_ms:.0f}ms, "
            f"{result.page_count} 页, "
            f"{len(result.text)} 字符"
        )
        return result
    except Exception as e:
        logger.warning(f"[解析] Docling 失败 ({doc_id})，回退到 Unstructured: {e}")

    # ===================================================================
    # 策略 2：Unstructured 回退解析
    # ===================================================================
    try:
        from src.parsers.unstructured_parser import UnstructuredParser

        parser = UnstructuredParser()
        result = parser.parse(file_path, doc_id)
        from src.parsers.text_cleaner import clean_text

        result.text = clean_text(result.text)
        logger.info(
            f"[解析] Unstructured 成功: {doc_id}, "
            f"耗时 {result.parse_time_ms:.0f}ms, "
            f"{result.page_count} 页"
        )
        return result
    except Exception as e:
        logger.error(f"[解析] Unstructured 也失败 ({doc_id}): {e}")
        raise RuntimeError(
            f"无法解析文档 {doc_id} ({file_path.name})。"
            f"Docling 和 Unstructured 均解析失败。"
        ) from e
