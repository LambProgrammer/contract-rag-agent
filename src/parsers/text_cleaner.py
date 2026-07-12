"""
文本清洗模块
============
在 RAG 系统中的角色：
    在文档解析（步骤 3）和条款分块（步骤 4）之间，清理解析输出的
    格式噪声，提高后续分块和检索的质量。

清洗项目及原因：
    1. 零宽字符（\\u200b 等）— 不可见但占用字符串长度，影响正则匹配和 embedding 质量
    2. 换页符（\\x0c / form feed）— PDF 分页残留，会被当作普通字符参与分词
    3. 全角数字（１２３）— 统一为半角以便 jieba 分词和编号匹配
    4. 冗余空行（连续 3+ 空行 → 2 空行）— 压缩无效空白
    5. 页眉页脚残留 — 部分 PDF 的页眉/页脚被 Docling 当成正文保留

注意：
    当前清洗为"保守模式"——只移除明确无用的噪声字符，不做语义层面的修改。
    不会删除任何合同条款正文内容。

用法：
    from src.parsers.text_cleaner import clean_text
    cleaned = clean_text(raw_markdown)
"""

import re

# 常见零宽字符
ZERO_WIDTH_CHARS = re.compile(r"[​‌‍\u200E\u200F﻿]")

# 换页符
FORM_FEED = re.compile(r"\x0c")

# 全角数字 → 半角数字
FULLWIDTH_DIGITS = str.maketrans(
    "０１２３４５６７８９",
    "0123456789",
)

# 连续 4+ 空行 → 2 空行
MULTIPLE_BLANK_LINES = re.compile(r"\n{4,}")

# 常见页眉页脚模式（独立成行的"第 X 页 / 共 Y 页"）
PAGE_HEADER_FOOTER = re.compile(
    r"^\s*(第\s*[0-9０-９]+\s*页\s*[／/]?\s*共\s*[0-9０-９]+\s*页)\s*$",
    re.MULTILINE,
)


def clean_text(text: str) -> str:
    """
    清洗解析输出的 Markdown 文本，移除格式噪声。

    参数：
        text: Docling/Unstructured 解析后的原始 Markdown 文本

    返回：
        清洗后的文本
    """
    # ① 零宽字符
    text = ZERO_WIDTH_CHARS.sub("", text)

    # ② 换页符
    text = FORM_FEED.sub("", text)

    # ③ 全角数字 → 半角
    text = text.translate(FULLWIDTH_DIGITS)

    # ④ 页眉页脚
    text = PAGE_HEADER_FOOTER.sub("", text)

    # ⑤ 冗余空行（最后执行，前面操作可能产生额外空行）
    text = MULTIPLE_BLANK_LINES.sub("\n\n", text)

    return text
