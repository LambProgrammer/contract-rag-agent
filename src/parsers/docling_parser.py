"""
Docling 解析器
=============
在 RAG 系统中的角色：
    主力文档解析器，负责将 PDF/Word 合同文件转换为 Markdown 结构化文本。
    Docling 是 IBM 开源的文档理解库，对 PDF 格式的支持远优于传统的 PyPDF2。

为什么选择 Docling：
    1. 深度理解文档结构：不只是提取文字，而是识别标题层级、段落边界、
       表格行列、列表项等语义结构
    2. Markdown 输出：保留合同条款的结构（标题 → ##，表格 → |---|---|），
       分块器可以利用这些标记做结构化切割
    3. 中英文混排：对中文合同（常有中英文术语混用）支持良好
    4. 表格提取：合同中的价目表、赔偿标准表等表格可以正确提取

Docling 2.x API 简介（学习参考）：
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert("合同.pdf")        # 返回 ConversionResult
    result.document.export_to_markdown()           # → Markdown 文本
    result.document.export_to_dict()               # → 结构化字典

局限性（为什么需要 Unstructured 回退）：
    - 纯扫描件 PDF（每页都是一张图片，没有文字层）需要 OCR，Docling 的 OCR 不如 Tesseract
    - 某些老旧的 .doc 格式（Word 97-2003）可能不支持
    - 损坏的 PDF 文件（XREF 表损坏）可能解析失败

用法：
    from src.parsers.docling_parser import DoclingParser
    parser = DoclingParser()
    result = parser.parse(Path("uploads/abc/合同.pdf"), doc_id="abc")
"""

import logging
import time
from pathlib import Path

from src.parsers import DocumentParseResult

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Docling 文档解析器。

    内部使用 Docling 的 DocumentConverter 将文件转换为结构化文档对象，
    然后导出为 Markdown。导出时会自动包含表格、列表、标题等语义结构。
    """

    def __init__(self) -> None:
        """
        初始化 Docling DocumentConverter。

        DocumentConverter 是 Docling 的核心入口，它的主要工作流程：
            1. 检测文档格式（PDF / DOCX / PPTX 等）
            2. 加载对应的后端引擎（PDF → pdfium，DOCX → python-docx）
            3. 构建文档模型（Document — 页 → 布局 → 文本/表格/图片）
            4. 应用 AI 模型增强（表格识别、阅读顺序检测等）
        """
        self._converter = None  # 延迟初始化，首次调用 parse() 时才加载

    @property
    def converter(self):
        """
        延迟加载 Docling DocumentConverter。

        为什么延迟加载：
            Docling 初始化时会加载 AI 模型到内存（约 200MB），
            如果 Worker 启动时就加载，每个 Worker 进程都占 200MB，
            内存很快耗尽。延迟加载意味着只有真正处理文档时才加载模型。

        线程安全说明：
            Celery 每个 Worker 是独立进程，不存在多线程竞争，
            所以这里不需要加锁。
        """
        if self._converter is None:
            logger.info("[Docling] 正在加载 DocumentConverter（首次初始化）...")
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
            logger.info("[Docling] DocumentConverter 加载完成")
        return self._converter

    def parse(self, file_path: Path, doc_id: str) -> DocumentParseResult:
        """
        解析文档文件，返回 Markdown 结构化文本。

        参数：
            file_path: 待解析文件的本地路径
            doc_id: 关联的文档 ID

        返回：
            DocumentParseResult：包含 Markdown 文本、页数、耗时等

        可能的错误：
            - FileNotFoundError: 文件不存在
            - ValueError: Docling 不支持此文件格式
            - RuntimeError: AI 模型加载失败（容器内首次运行时可能缺少模型文件）
        """
        logger.info(f"[Docling] 开始解析: {file_path.name}")

        start_time = time.perf_counter()

        # ---- 核心 API 调用：DocumentConverter.convert() ----
        # 这是 Docling 整个解析流程的入口，内部会：
        #   1. 自动选择后端（PDF → pdfium, DOCX → python-docx）
        #   2. 识别页面布局（段落、表格、图片区域）
        #   3. 提取文本并保持阅读顺序
        conversion_result = self.converter.convert(str(file_path))

        # ---- 导出为 Markdown ----
        # export_to_markdown() 会：
        #   - 将标题格式化（一级标题 → #, 二级 → ##）
        #   - 将表格转为 Markdown table（| col1 | col2 |）
        #   - 保留列表和缩进
        markdown_text = conversion_result.document.export_to_markdown()

        # ---- 获取页数 ----
        # Docling 内部按页组织内容，pages 包含了每一页的文本/布局信息
        page_count = (
            len(conversion_result.document.pages)
            if hasattr(conversion_result.document, "pages")
            else 0
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[Docling] 解析完成: {file_path.name}, "
            f"{page_count} 页, {len(markdown_text)} 字符, {elapsed_ms:.0f}ms"
        )

        return DocumentParseResult(
            doc_id=doc_id,
            text=markdown_text,
            page_count=page_count,
            parse_time_ms=elapsed_ms,
            parser_used="docling",
        )
