"""
合同比对核心分析器
==================
在 RAG 系统中的角色：
    实现跨合同比对的全部核心逻辑：
    B0: 全文语义相关性预检——拒绝"完全不相关"合同的比对，避免产出误导报告
    B1: 条款对齐——正则 clause_ref 匹配 + dense 向量语义回退，解决编号体系不同的对齐问题
    B2: 相同内容预过滤——MD5 hash 秒杀完全相同条款，跳过 LLM 调用，节省 token 和延迟
    B3: 上下文扩充 + LLM 逐句对比——前后各 1 个相邻 Chunk 作为背景，LLM 做逐句 diff + 风险评估

数据流：
    doc_a_chunks, doc_b_chunks
      → B0: 全文相关性检查 → 不相关则拒绝
      → B1: 条款对齐 → aligned_pairs + orphans
      → B2: MD5 预过滤 → to_llm（需对比的）+ identical（一致的）
      → B3: 上下文扩充 → LLM 对比 → 差异列表

用法：
    from src.comparator.diff_analyzer import compare_documents
    result = compare_documents(chunks_a, chunks_b)
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.chunkers.legal_chunker import Chunk
from src.core.config import settings

logger = logging.getLogger(__name__)

# 全文相似度阈值——从 .env 读取（可配置），低于此值拒绝比对
# 语义对齐阈值——dense 向量 cosine 低于此值判定为无匹配
SEMANTIC_ALIGN_THRESHOLD = 0.7


# ============================================================
# 数据模型
# ============================================================
@dataclass
class AlignedPair:
    """一对对齐成功的条款"""

    chunk_a: Chunk
    chunk_b: Chunk
    align_method: str  # "regex" | "semantic"


@dataclass
class DiffItem:
    """单条对比差异"""

    clause_ref_a: str
    clause_ref_b: str
    align_method: str
    is_identical: bool
    llm_analysis: str = ""


@dataclass
class CompareStats:
    """比对全局统计"""

    total_aligned: int
    identical: int
    with_differences: int
    only_in_a: int
    only_in_b: int


@dataclass
class CompareResult:
    """比对完整结果"""

    doc_id_a: str
    doc_id_b: str
    relevance_score: float
    stats: CompareStats
    differences: List[DiffItem]
    only_in_a: List[Chunk]
    only_in_b: List[Chunk]


# ============================================================
# B0: 全文相关性预检
# ============================================================
def check_relevance(chunks_a: List[Chunk], chunks_b: List[Chunk]) -> Tuple[float, bool]:
    """
    计算两份合同全文的语义相似度，判断是否值得比对。

    防止用户将完全不相关的合同放入比对（如"房屋租赁"vs"软件开发"），
    此时 clause_ref 几乎全部对不上，输出大量"缺失"条目将误导用户。

    全文编码实时进行（仅 2 次 encode，~100ms），无需预存索引。
    """
    from src.indexing.embedder import BGEEmbedder

    text_a = "\n".join(c.text for c in chunks_a)
    text_b = "\n".join(c.text for c in chunks_b)

    if not text_a.strip() or not text_b.strip():
        return 0.0, False

    embedder = BGEEmbedder()
    vecs = embedder.encode([text_a, text_b])

    import numpy as np

    dot = np.dot(vecs[0], vecs[1])
    norm = np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
    similarity = float(dot / norm) if norm > 0 else 0.0

    is_relevant = similarity >= settings.compare_relevance_threshold
    logger.info(
        f"[B0] similarity={similarity:.4f}, "
        f"threshold={settings.compare_relevance_threshold}, relevant={is_relevant}"
    )
    return similarity, is_relevant


# ============================================================
# B1: 条款对齐
# ============================================================
def align_clauses(
    chunks_a: List[Chunk], chunks_b: List[Chunk]
) -> Tuple[List[AlignedPair], List[Chunk], List[Chunk]]:
    """
    对两份合同的 Chunks 做条款级对齐。

    策略：
        ① clause_ref 正则匹配（字符串完全相同 + 前 8 字符模糊容错）
        ② 未匹配的 → dense 向量 cosine 回退对齐
        ③ 仍未匹配 → orphans（L1 "缺失/新增"）
    """
    aligned_pairs = []
    unmatched_a = []
    unmatched_b = list(chunks_b)

    for chunk_a in chunks_a:
        match = _find_by_clause_ref(chunk_a, unmatched_b)
        if match:
            aligned_pairs.append(
                AlignedPair(chunk_a=chunk_a, chunk_b=match, align_method="regex")
            )
            unmatched_b.remove(match)
        else:
            unmatched_a.append(chunk_a)

    # ---- 语义回退：对 regex 未匹配的 A 条款做向量对齐 ----
    if unmatched_a and unmatched_b:
        embedder = None
        still_unmatched_a = []

        for chunk_a in unmatched_a:
            if not unmatched_b:
                still_unmatched_a.append(chunk_a)
                continue

            if embedder is None:
                from src.indexing.embedder import BGEEmbedder

                embedder = BGEEmbedder()

            texts_b = [c.text for c in unmatched_b]
            vec_a = embedder.encode([chunk_a.text])[0]
            vecs_b = embedder.encode(texts_b)

            import numpy as np

            dots = np.dot(vec_a, vecs_b.T)
            norms = np.linalg.norm(vec_a) * np.linalg.norm(vecs_b, axis=1)
            scores = dots / np.maximum(norms, 1e-8)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= SEMANTIC_ALIGN_THRESHOLD:
                aligned_pairs.append(
                    AlignedPair(
                        chunk_a=chunk_a,
                        chunk_b=unmatched_b[best_idx],
                        align_method="semantic",
                    )
                )
                unmatched_b.pop(best_idx)
            else:
                still_unmatched_a.append(chunk_a)

        unmatched_a = still_unmatched_a

    logger.info(
        f"[B1] {len(aligned_pairs)} 对匹配 "
        f"({sum(1 for p in aligned_pairs if p.align_method == 'regex')} regex, "
        f"{sum(1 for p in aligned_pairs if p.align_method == 'semantic')} semantic), "
        f"A 独有 {len(unmatched_a)}, B 独有 {len(unmatched_b)}"
    )

    return aligned_pairs, unmatched_a, unmatched_b


def _find_by_clause_ref(chunk_a: Chunk, candidates: List[Chunk]) -> Optional[Chunk]:
    """按 clause_ref 字符串精确匹配，无意义的标签跳过（交由语义回退处理）"""
    ref_a = chunk_a.clause_ref.strip()
    if not ref_a or ref_a in ("未命名条款", "前言"):
        return None
    for c in candidates:
        ref_b = c.clause_ref.strip()
        if not ref_b or ref_b in ("未命名条款", "前言"):
            continue
        if ref_a == ref_b:
            return c
    return None


# ============================================================
# B2: 预过滤相同内容
# ============================================================
def filter_identical(
    pairs: List[AlignedPair],
) -> Tuple[List[AlignedPair], List[AlignedPair]]:
    """
    用 MD5 hash 过滤完全相同的条款对，避免浪费 LLM token。

    rapid_hash = hashlib.md5(text.encode()).hexdigest()
    相同文本产生相同 hash → 秒杀判定为 identical → 跳过 LLM 调用。
    """
    to_llm = []
    identical = []
    for pair in pairs:
        hash_a = hashlib.md5(pair.chunk_a.text.encode("utf-8")).hexdigest()
        hash_b = hashlib.md5(pair.chunk_b.text.encode("utf-8")).hexdigest()
        if hash_a == hash_b:
            identical.append(pair)
        else:
            to_llm.append(pair)

    logger.info(
        f"[B2] {len(identical)} 对完全相同（跳过）, {len(to_llm)} 对需 LLM 对比"
    )
    return to_llm, identical


# ============================================================
# B3: 上下文扩充 + LLM 对比
# ============================================================
def llm_diff(
    to_llm: List[AlignedPair],
    all_chunks_a: List[Chunk],
    all_chunks_b: List[Chunk],
) -> List[DiffItem]:
    """
    对需要对比的条款对，扩充上下文后送 LLM 做逐句 diff + 风险评估。

    上下文窗口：前后各 1 个相邻 Chunk（window=1），
    确保 LLM 看到条款的定义、适用条件等背景信息，避免孤立判断失真。
    """
    from langchain_deepseek import ChatDeepSeek
    from pydantic import SecretStr

    items = []
    if not to_llm:
        return items

    llm = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=SecretStr(settings.deepseek_api_key),
        temperature=0.2,
    )

    for idx, pair in enumerate(to_llm, 1):
        logger.info(
            f"[B3] ({idx}/{len(to_llm)}) "
            f"{pair.chunk_a.clause_ref} vs {pair.chunk_b.clause_ref}"
        )

        ctx_a = _expand_context(pair.chunk_a, all_chunks_a, window=1)
        ctx_b = _expand_context(pair.chunk_b, all_chunks_b, window=1)

        analysis = _call_llm_diff(llm, ctx_a, ctx_b)
        items.append(
            DiffItem(
                clause_ref_a=pair.chunk_a.clause_ref,
                clause_ref_b=pair.chunk_b.clause_ref,
                align_method=pair.align_method,
                is_identical=False,
                llm_analysis=analysis,
            )
        )

    return items


def _expand_context(chunk: Chunk, all_chunks: List[Chunk], window: int = 1) -> str:
    """取 Chunk 前后各 window 个相邻块，拼接为上下文"""
    idx = chunk.chunk_index
    neighbors = (
        all_chunks[max(0, idx - window) : idx]
        + [chunk]
        + all_chunks[idx + 1 : idx + 1 + window]
    )
    return "\n\n".join(c.text for c in neighbors)


def _call_llm_diff(llm, ctx_a: str, ctx_b: str) -> str:
    """调用 LLM 做逐句 diff"""
    prompt = (
        "你是一个合同对比分析专家。请逐句对比以下两份合同条款。\n\n"
        "【甲方版本】\n"
        f"{ctx_a}\n\n"
        "【乙方版本】\n"
        f"{ctx_b}\n\n"
        "请按以下格式回答：\n"
        "1. 逐条列出措辞差异（原文对比格式：'甲方: ... → 乙方: ...'）\n"
        "2. 每条差异的风险评估——是否可能导致：\n"
        "   - 权利义务不对等\n"
        "   - 责任转移\n"
        "   - 时间或金额窗口变化\n"
        "   - 履约标准改变\n"
        "3. 如果条款完全相同，写'无差异'\n"
    )
    response = llm.invoke(prompt)
    return str(response.content)


# ============================================================
# 主入口：完整比对
# ============================================================
def compare_documents(
    doc_id_a: str,
    doc_id_b: str,
    chunks_a: List[Chunk],
    chunks_b: List[Chunk],
) -> CompareResult:
    """执行完整跨合同比对（B0→B1→B2→B3）"""
    # B0
    relevance, is_relevant = check_relevance(chunks_a, chunks_b)
    if not is_relevant:
        raise ValueError(
            f"两份合同的全文语义相似度仅为 {relevance:.2f}（阈值 {settings.compare_relevance_threshold}），"
            f"主题差异过大，无法进行有效比对。请确认两份合同属于同一类型。"
        )

    # B1
    aligned, orphans_a, orphans_b = align_clauses(chunks_a, chunks_b)

    # B2
    to_llm, identical_pairs = filter_identical(aligned)

    # B3
    diff_items = llm_diff(to_llm, chunks_a, chunks_b)

    for pair in identical_pairs:
        diff_items.append(
            DiffItem(
                clause_ref_a=pair.chunk_a.clause_ref,
                clause_ref_b=pair.chunk_b.clause_ref,
                align_method=pair.align_method,
                is_identical=True,
                llm_analysis="条款内容完全一致",
            )
        )

    stats = CompareStats(
        total_aligned=len(aligned),
        identical=len(identical_pairs),
        with_differences=len(to_llm),
        only_in_a=len(orphans_a),
        only_in_b=len(orphans_b),
    )

    return CompareResult(
        doc_id_a=doc_id_a,
        doc_id_b=doc_id_b,
        relevance_score=round(relevance, 4),
        stats=stats,
        differences=diff_items,
        only_in_a=orphans_a,
        only_in_b=orphans_b,
    )
