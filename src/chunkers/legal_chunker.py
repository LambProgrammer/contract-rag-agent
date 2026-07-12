"""
合同条款结构化分块器（Legal Clause Chunker）
===========================================
在 RAG 系统中的角色：
    将步骤 3 解析出的长 Markdown 文本，按合同条款编号切割为语义完整的"块"。
    这是 RAG 管道中决定检索精度的关键一步——分块方式直接决定了
    "用户问一个条款，系统能否精确召回对应的那一条"。

为什么不能用简单的固定长度切割：
    合同文本有天然的编号结构（"第一条"、"1.1"、"（一）"等），
    每个编号对应一个完整的语义单元。如果按每 500 字硬切：
        ┌─────────────────────────────┐
        │ ...违约方应赔偿守约方因此遭   │ ← 前半句在 chunk A
        │ 受的全部直接损失，包括但不限   │
        ├─────────────────────────────┤ ← 切分线穿过一个完整条款
        │ 于律师费、诉讼费、差旅费等... │ ← 后半句在 chunk B
        └─────────────────────────────┘
    检索时用户问"违约赔偿范围"，向量检索可能只召回 chunk A，
    LLM 拿到半句话，无法给出准确答案。

结构化分块的好处：
    - 每个块是一个完整的条款语义单元（边界 = 法律条款边界）
    - 检索召回的是整条条款，不截断不遗漏
    - 可附带条款引用（"第五条第三款"），提升回答可信度

分块策略（三步走）：
    ① 按条款编号 / 附件标题正则切割 → 主切分点
    ② 合并过短的块（相邻 < 100 字的合并，避免孤立的标题行）
    ③ 拆分过长的块（单块 > 1500 字，降级到子条款编号再次切割）

配置来源：
    MIN_CHUNK_SIZE / MAX_CHUNK_SIZE 从 .env 读取（见 src/core/config.py）

用法：
    from src.chunkers.legal_chunker import LegalClauseChunker
    chunker = LegalClauseChunker()
    chunks = chunker.chunk(markdown_text, doc_id="abc123")
    for c in chunks:
        print(c.clause_ref, "→", c.text[:50])
"""

import logging
import re
import uuid
from dataclasses import dataclass
from typing import List

from src.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================
@dataclass
class Chunk:
    """一个经过分块处理的文档片段。

    每个 Chunk 代表合同中的一个"语义单元"——通常是一条完整的条款。
    步骤 5 会把每个 Chunk 向量化后写入 Qdrant。
    """

    chunk_id: str
    """块唯一 ID（UUID4），Qdrant 中作为 point id"""

    doc_id: str
    """来源文档 ID，用于追溯和过滤"""

    text: str
    """块的完整文本内容（Markdown 格式）"""

    chunk_index: int
    """块在文档中的序号（0-based），保持原文顺序"""

    clause_ref: str
    """条款引用标识，如 "第一条"、"3.1"、"附件一"，用于答案引用"""

    start_char: int
    """块文本在原始文档中的起始字符位置（用于溯源）"""

    end_char: int
    """块文本在原始文档中的结束字符位置"""


# ============================================================
# 正则模式：匹配合同条款编号与附件标题
# ============================================================
# 以下模式按"切割优先级"从高到低排列。
# 匹配到的编号将被用作 clause_ref，切割点就是编号出现的位置。

# ----------------------------------------------------------
# L1-A（顶级切分点）："附件X"
# 合同末尾的附件/附录通常是最重要的内容之一（价目表、服务清单等），
# 必须作为顶级切分点，不能黏在最后一条条款里
# 匹配格式：
#   ## 附件一：服务费用明细
#   ## 附件A：付款计划
#   # 附件1：合同清单
#   ### 附件三 验收标准
# 支持：中文数字（一～十）、阿拉伯数字（1-99）、大写字母（A-Z）
# ----------------------------------------------------------
PATTERN_APPENDIX = re.compile(
    r"^#{1,4}\s*附件"
    r"(?:[一二三四五六七八九十百千万]+"  # 中文数字
    r"|[A-Z]"  # 大写字母
    r"|[1-9]\d{0,1}"  # 阿拉伯数字 1-99
    r")",
    re.MULTILINE,
)

# L1-B（顶级切分点）："第X章"、"第X节"、"第X条"、"第X款"
# 匹配：第一章、第二节、第三条、第五款 等
PATTERN_CHAPTER_SECTION = re.compile(
    r"^#{1,4}\s*第[一二三四五六七八九十百千万\d]+[章节条款]",
    re.MULTILINE,
)

# L2（主切分点）：Markdown 标题 + 条款关键词
# 匹配："## 第一条 定义"、"### 2.1 服务范围"
PATTERN_HEADING_CLAUSE = re.compile(
    r"^#{1,4}\s*(?:第[一二三四五六七八九十百千万\d]+[条款节章]|\d+(?:\.\d+)*)\s",
    re.MULTILINE,
)

# L3（主切分点）：纯文本条款编号
# 匹配："第一条  "、"第二条  "、 "第十条  "
# 注意：中文数字后跟"条"是合同最核心的编号体系
PATTERN_PLAIN_CLAUSE = re.compile(
    r"(?:^|\n)\s*第[一二三四五六七八九十百千万\d]+条[\s　]+",
)

# L4（子条款拆分）：用数字编号（当单块过长时使用）
# 匹配："一、"、"（一）"、"1."、"1.1 "、"1.1.1 " 等
PATTERN_SUB_CLAUSE = re.compile(
    r"(?:^|\n)\s*(?:[（(]?\s*[一二三四五六七八九十\d]+[、)）.．]\s*|\d+(?:\.\d+)+(?:\s|．))",
)


# ============================================================
# 分块器
# ============================================================
class LegalClauseChunker:
    """
    合同专用结构化分块器。

    按中文法律文书的标准编号体系，将长文本切割为语义完整的条款块。
    输入：Markdown 文本
    输出：Chunk 列表（每个 Chunk 是一条完整的合同条款或附件章节）
    """

    def __init__(
        self,
        min_chunk_size: int | None = None,
        max_chunk_size: int | None = None,
    ) -> None:
        """
        初始化分块器。

        参数：
            min_chunk_size: 最小块字符数（低于此会与相邻块合并），
                           默认从 settings.MIN_CHUNK_SIZE 读取（100）
            max_chunk_size: 最大块字符数（超过此会拆分为子条款），
                           默认从 settings.MAX_CHUNK_SIZE 读取（1500）
        """
        self.min_size = min_chunk_size or settings.min_chunk_size
        self.max_size = max_chunk_size or settings.max_chunk_size

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        对文档文本执行结构化分块。

        三步策略：
            ① 按条款编号 / 附件标题正则切割（主切分点）
            ② 合并过短的块
            ③ 拆分过长的块

        参数：
            text: 原始 Markdown 文本（步骤 3 的输出）
            doc_id: 关联的文档 ID

        返回：
            Chunk 列表，按原文顺序排列
        """
        logger.info(f"[分块] 开始处理文档 {doc_id}，原文 {len(text)} 字符")

        if not text.strip():
            return []

        # ① 主切割
        raw_chunks = self._split_by_clause_patterns(text)
        logger.info(f"[分块] 步骤① 主切割: {len(raw_chunks)} 个块")

        # ② 合并过短块
        merged_chunks = self._merge_short_chunks(raw_chunks)
        logger.info(f"[分块] 步骤② 合并过短: {len(merged_chunks)} 个块")

        # ③ 拆分过长块
        final_chunks = self._split_long_chunks(merged_chunks)
        logger.info(f"[分块] 步骤③ 拆分过长: {len(final_chunks)} 个块")

        # 转为 Chunk 对象
        result = self._build_chunk_objects(final_chunks, doc_id)
        logger.info(
            f"[分块] 完成: {len(result)} 个 Chunk，"
            f"平均 {sum(len(c.text) for c in result) / len(result):.0f} 字/块"
        )
        return result

    # ----------------------------------------------------------
    # 步骤①：按层级模式切割
    # ----------------------------------------------------------
    def _split_by_clause_patterns(self, text: str) -> List[str]:
        """
        按条款编号正则逐级尝试切割。

        切割策略（优先级从高到低）：
            L1-A: "附件一/附件A/附件1" → 顶级切分点，附件内容独立成块
            L1-B: "第X章/节/条/款" 标题形式 → 顶级切分点
            L2:   Markdown 标题 + 条款关键词
            L3:   纯文本 "第X条"
            L4:   子条款编号（仅在后续拆分过长块时使用）

        如果以上都没匹配到，说明文档没有标准条款编号体系，
        则回退到按 Markdown 标题切割（## / ###）。
        """
        # 将 L1-A（附件）+ L1-B（章/节/条）+ L2 + L3 合并为主切分模式
        # 注意：附件放在最前面，确保"附件一"不会被"第一条"的 L3 模式错误匹配
        main_pattern = re.compile(
            # L1-A: 附件标题（顶级优先级）
            r"(?:^#{1,4}\s*附件"
            r"(?:[一二三四五六七八九十百千万]+"  # 中文数字
            r"|[A-Z]"  # 大写字母
            r"|[1-9]\d{0,1})"  # 阿拉伯数字 1-99
            r")"
            r"|"
            # L1-B: "第X章/节/条/款" 标题
            r"(?:^#{1,4}\s*第[一二三四五六七八九十百千万\d]+[章节条款])"
            r"|"
            # L2: Markdown 标题 + 条款关键词
            r"(?:^#{1,4}\s*(?:第[一二三四五六七八九十百千万\d]+[条款节章]|\d+(?:\.\d+)*)\s)"
            r"|"
            # L3: 纯文本 "第X条"
            r"(?:^|\n)\s*第[一二三四五六七八九十百千万\d]+条[\s　]+"
            r"|"
            # 回退: Markdown 标题（任何 ## 级别的标题）
            r"(?:^|\n)(?=#{1,4}\s)",
            re.MULTILINE,
        )

        # 找到所有切分点位置
        split_positions = [m.start() for m in main_pattern.finditer(text)]

        if not split_positions:
            # 完全没有匹配到任何条款编号 → 整个文档作为一个块
            return [text.strip()] if text.strip() else []

        # 如果第一个条款编号不在文本开头，把前面的内容单独作为"前言"块
        if split_positions[0] > 0:
            split_positions.insert(0, 0)

        # 按位置切割
        chunks = []
        for i, pos in enumerate(split_positions):
            end = split_positions[i + 1] if i + 1 < len(split_positions) else len(text)
            chunk_text = text[pos:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    # ----------------------------------------------------------
    # 步骤②：合并相邻的过短块
    # ----------------------------------------------------------
    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """
        合并过短的相邻块，确保每个块至少达到 min_size 字符。

        为什么需要合并：
            - 条款标题（"## 第一条 定义"）通常只有十几个字，
              单独作为一个块，vector 语义信息不够丰富，检索效果差
            - 应把标题 + 紧跟的条款正文合并，形成一个有意义的完整片段

        合并策略（贪心）：
            从前往后扫描，如果当前块 < min_size，
            就把它合并到前一个块（如果前一个存在）或后一个块。
        """
        if len(chunks) <= 1:
            return chunks

        result = []
        buffer = ""

        for chunk in chunks:
            if len(chunk) < self.min_size and result:
                # 当前块太短 → 追加到上一个块的末尾
                result[-1] = result[-1] + "\n\n" + chunk
            elif len(chunk) < self.min_size and not result:
                # 第一个块就太短 → 暂存到 buffer，等下个块来一起合并
                buffer = chunk
            elif buffer:
                # 上一个块太短且被 buffer 暂存了 → 合并后添加到结果
                result.append(buffer + "\n\n" + chunk)
                buffer = ""
            else:
                result.append(chunk)

        # 如果最后还剩一个过短的 buffer，附加到最后一个结果
        if buffer:
            if result:
                result[-1] = result[-1] + "\n\n" + buffer
            else:
                result.append(buffer)

        return result

    # ----------------------------------------------------------
    # 步骤③：拆分过长的块
    # ----------------------------------------------------------
    def _split_long_chunks(self, chunks: List[str]) -> List[str]:
        """
        拆分超过 max_size 的块，降级到子条款编号切割。

        例如一个"第一条 定义"包含了 2000 字的定义列表，
        则按"（一）"、"（二）"等子编号拆成几个子块。
        """
        result = []
        for chunk in chunks:
            if len(chunk) <= self.max_size:
                result.append(chunk)
            else:
                # 块太长 → 用子条款模式再切一次
                sub_chunks = self._split_by_sub_pattern(chunk)
                result.extend(sub_chunks)
        return result

    def _split_by_sub_pattern(self, text: str) -> List[str]:
        """
        用子条款编号模式再切割。

        子条款模式示例：
            "一、定义"、"（一）服务范围"、"1. 费用构成"、"1.1 支付方式"
        """
        matches = list(PATTERN_SUB_CLAUSE.finditer(text))

        if not matches:
            # 找不到子条款编号 → 无法拆分，只能保持原样
            logger.warning(
                f"[分块] 块过长 ({len(text)} 字) 但无子条款编号可拆分，保持原样"
            )
            return [text]

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    # ----------------------------------------------------------
    # 构建 Chunk 对象
    # ----------------------------------------------------------
    def _build_chunk_objects(self, texts: List[str], doc_id: str) -> List[Chunk]:
        """
        将切割后的文本列表转为 Chunk 数据对象。

        提取每个块的：
            - clause_ref: 从文本开头提取条款编号或附件标题
            - start_char / end_char: 在原文中的位置
        """
        chunks = []
        cursor = 0
        for i, text in enumerate(texts):
            clause_ref = self._extract_clause_ref(text)
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=text,
                chunk_index=i,
                clause_ref=clause_ref,
                start_char=cursor,
                end_char=cursor + len(text),
            )
            chunks.append(chunk)
            cursor += len(text) + 2  # +2 补偿合并时的 "\n\n" 分隔符
        return chunks

    def _extract_clause_ref(self, text: str) -> str:
        """
        从块文本中提取条款引用标识。

        提取优先级：
            1. "附件X" → 返回 "附件一" / "附件A" / "附件2" 等
            2. "第X条" / "第X章" → 返回完整标题首行
            3. 数字编号（如 3.1）→ 返回标题首行
            4. 无编号 → "前言" / "未命名条款"

        示例：
            "## 附件一：服务费用明细" → "附件一：服务费用明细"
            "## 第一条 违约责任\n..." → "第一条 违约责任"
            "### 3.1 费用结算方式\n..." → "3.1 费用结算方式"
            "本合同自双方签字之日起..." → "前言"
        """
        first_line = text.split("\n")[0].strip()
        # 去掉 Markdown 标题标记（### 等）
        first_line = re.sub(r"^#+\s*", "", first_line)

        # ---- 检查附件标题 ----
        # 匹配：附件一、附件A、附件1（及其后续说明文字）
        appendix_match = re.match(
            r"(附件"
            r"(?:[一二三四五六七八九十百千万]+"  # 中文数字
            r"|[A-Z]"  # 大写字母
            r"|[1-9]\d{0,1}"  # 阿拉伯数字
            r")[：:\s]*(.*))",
            first_line,
        )
        if appendix_match:
            # 返回完整附件引用，如 "附件一：服务费用明细"
            ref = appendix_match.group(1)
            return ref[:50] if len(ref) > 50 else ref

        # ---- 检查条款编号 ----
        # "第X条"、"第X章"、"第X款"
        clause_match = re.match(
            r"(第[一二三四五六七八九十百千万\d]+[章节条款])",
            first_line,
        )
        if clause_match:
            return first_line[:50]

        # ---- 检查数字编号（如 3.1、2.3.1）----
        num_match = re.match(r"(\d+(?:\.\d+)+)", first_line)
        if num_match:
            return first_line[:50]

        # ---- 回退：检查正文中是否包含条款编号 ----
        fallback = re.search(r"第[一二三四五六七八九十百千万\d]+[条款章]", text)
        if fallback:
            return fallback.group(0)

        # ---- 最终回退 ----
        # 如果第一行看起来像标题（较短），用标题名
        if first_line and 2 < len(first_line) <= 30:
            return first_line
        # 块的开头若不是编号而是一般叙述文字，大概率是合同前言部分
        if len(text) > 0 and not re.match(
            r"(第|附件|[一二三四五六七八九十\d]+[.)）])", text.lstrip()
        ):
            return "前言"

        # 取文本前 30 个字符作为 clause_ref（比"未命名条款"有辨识度）
        return text.strip()[:30] if text.strip() else "未命名条款"
