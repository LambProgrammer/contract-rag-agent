"""
Unstructured 回退解析器
=======================
在 RAG 系统中的角色：
    当 Docling 解析 PDF/Word 失败时，作为备用方案接管解析任务。
    Unstructured 是一个专门为 RAG 场景设计的文档预处理库，
    对难以处理的文档格式（老旧 .doc、扫描件 OCR、损坏的 PDF）有更好的容错性。

为什么不能只用 Docling：
    - Docling 对文件格式有严格要求（PDF 必须符合规范）
    - 实际业务中的合同文件来源多样（扫描件、旧版 Word、格式损坏的 PDF）
    - 需要一条"保底"路径确保系统不会因为文件格式问题而完全不可用

Unstructured 的 partition() 函数做了什么：
    1. 自动检测文件类型（PDF / DOCX / DOC / TXT / HTML / 图片等）
    2. 根据类型选择"分区策略"：
       - PDF: 先用 pdfminer 提取文字，失败则用 OCR（Tesseract）
       - DOCX: 用 python-docx 读 Office Open XML
       - DOC: 用 libreoffice 转 PDF 再解析（容器需预装 libreoffice，较重）
    3. 将文档拆分为 Element 列表（Title / NarrativeText / ListItem / Table）
    4. 每个 Element 包含：文本内容 + 元数据（页码、位置坐标）

与 Docling 的区别（学习参考）：
    | 维度       | Docling                | Unstructured            |
    |-----------|------------------------|-------------------------|
    | 结构理解   | 深度（文档树模型）      | 浅层（分区 + 元素类型）  |
    | OCR 能力  | 基础                   | Tesseract 集成，更强     |
    | 输出格式   | Markdown / JSON        | Element 列表             |
    | 速度       | 快（AI 模型优化）       | 较慢（尤其是 OCR 模式）   |
    | 中文支持   | 好（多语言模型）        | 一般（依赖 Tesseract）    |

用法：
    from src.parsers.unstructured_parser import UnstructuredParser
    parser = UnstructuredParser()
    result = parser.parse(Path("uploads/abc/旧合同.doc"), doc_id="abc")
"""

import logging
import time
from pathlib import Path

from src.parsers import DocumentParseResult

logger = logging.getLogger(__name__)


class UnstructuredParser:
    """
    Unstructured 回退文档解析器。

    使用 Unstructured 的 partition() 函数提取文档结构化元素，
    并拼接为保留段落结构的纯文本。
    """

    def parse(self, file_path: Path, doc_id: str) -> DocumentParseResult:
        """
        使用 Unstructured 解析文档文件。

        参数：
            file_path: 待解析文件的本地路径
            doc_id: 关联的文档 ID

        返回：
            DocumentParseResult：包含提取的文本和元数据
        """
        logger.info(f"[Unstructured] 开始回退解析: {file_path.name}")

        start_time = time.perf_counter()

        # ---- 核心 API 调用：partition() ----
        # partition() 是 Unstructured 的统一入口函数。
        #
        # 参数说明：
        #   filename: 文件路径（Unstructured 根据扩展名自动选策略）
        #   strategy: 分区策略
        #     - "auto" (默认): 自动选最优策略
        #     - "fast": 只用 pdfminer/python-docx，速度快但不支持 OCR
        #     - "hi_res": 高精度模式（含 OCR），慢但覆盖更多格式
        #     - "ocr_only": 纯 OCR 模式，用于纯扫描件
        #   include_page_breaks: 在文本中插入分页标记
        #
        from unstructured.partition.auto import partition

        elements = partition(
            filename=str(file_path),
            strategy="auto",  # 让 Unstructured 自动选择策略
            include_page_breaks=True,  # 保留分页信息，方便定位原文
        )

        # ---- 拼接 Element 为文本 ----
        # Unstructured 返回的是 Element 列表，每个 Element 代表文档中的一个语义单元
        # 例如：一个段落 → NarrativeText，一个标题 → Title，一个列表项 → ListItem
        #
        # 这里用双换行拼接，让段落之间有空行，保持可读性
        # 后续分块器可以按空行做初步切割
        text = "\n\n".join(str(el) for el in elements)

        # ---- 估算页数 ----
        # Unstructured 的 Element 上带有 metadata.page_number 属性
        # 取最大的页码作为总页数估计
        page_count = 0
        for el in elements:
            page_num = (
                getattr(el.metadata, "page_number", None)
                if hasattr(el, "metadata")
                else None
            )
            if page_num and page_num > page_count:
                page_count = page_num

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[Unstructured] 回退解析完成: {file_path.name}, "
            f"{page_count} 页, {len(text)} 字符, {elapsed_ms:.0f}ms"
        )

        return DocumentParseResult(
            doc_id=doc_id,
            text=text,
            page_count=page_count,
            parse_time_ms=elapsed_ms,
            parser_used="unstructured",
        )
