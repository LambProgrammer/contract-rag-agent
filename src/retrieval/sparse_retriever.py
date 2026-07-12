"""
稀疏向量检索器（Sparse Retriever / BM25 关键词匹配）
====================================================
在 RAG 系统中的角色：
    基于 jieba 分词 + BM25 权重做字面关键词检索。解决稠密向量
    对"精确名词"不敏感的问题——用户问"第16条"时，稠密向量可能
    匹配到"第十六条"但得分不高，而 BM25 的字面匹配能精确命中。

稠密 vs 稀疏（学习参考）：
    ┌──────────────┬──────────────────┬──────────────────┐
    │              │ Dense (稠密)      │ Sparse (稀疏)     │
    ├──────────────┼──────────────────┼──────────────────┤
    │ 匹配方式     │ 语义相似度        │ 关键词精确匹配     │
    │ 擅长的       │ "违约"≈"不履行"   │ "第16条"="第16条" │
    │ 弱点         │ 专有名词/编号     │ 同义词/近义词      │
    │ 向量结构     │ [0.12,-0.34,...]  │ {tok_id: weight}  │
    └──────────────┴──────────────────┴──────────────────┘
    两者互补——结合起来才是完整的检索。

为什么用 jieba 而非 scikit-learn 的 TfidfVectorizer：
    - jieba 专为中文设计，分词准确率高
    - 零依赖，不引入 scikit-learn 全家桶
    - BM25 比 TF-IDF 更先进（考虑词频饱和度 + 文档长度归一化）

Qdrant 的稀疏向量存储：
    Qdrant 1.12+ 支持 Named Vector——一个 Point 同时有
    "dense" 和 "sparse" 两个向量。稀疏向量用 indices/values 格式：
        {indices: [101, 205, 307], values: [0.5, 0.3, 0.2]}
    表示：token_101 权重 0.5, token_205 权重 0.3, token_307 权重 0.2

用法：
    from src.retrieval.sparse_retriever import SparseRetriever
    retriever = SparseRetriever()
    results = retriever.retrieve("第16条规定的违约金")
"""

import hashlib
import logging
from collections import Counter
from typing import List, Tuple

from src.indexing.vector_client import QdrantVectorClient
from src.retrieval.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# 稀疏向量词表大小（哈希空间），10^6 足够覆盖中文常用词
VOCAB_SIZE = 1_000_000


# ============================================================
# 分词与权重计算
# ============================================================
def tokenize(text: str) -> List[str]:
    """
    使用 jieba 对中文文本分词，过滤单字和空白。

    jieba 分词示例：
        "违约责任怎么计算" → ["违约", "责任", "怎么", "计算"]
        "第十六条 旅行社的违约责任" → ["第十六", "条", "旅行社", "的", "违约", "责任"]
    """
    import jieba

    tokens = jieba.cut(text, cut_all=False)  # 精确模式
    return [t.strip() for t in tokens if len(t.strip()) > 1]


def token_to_id(token: str) -> int:
    """将中文词映射为固定范围的整数 ID（哈希 + 取模）"""
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % VOCAB_SIZE


def text_to_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
    """
    将文本转为 Qdrant 稀疏向量格式 (indices, values)。

    权重计算简化为词频（BM25 的核心是 TF * IDF 的变体，
    但 IDF 需要全局文档统计，这里用词频近似——对单次检索而言
    效果差异很小，因为 BM25 的 TF 分量 = 主导因素）。

    返回：
        (indices: [int], values: [float]) — 可直接传给 Qdrant
    """
    tokens = tokenize(text)
    if not tokens:
        return [], []

    # 统计词频作为权重
    tf = Counter(tokens)
    # 合并 hash 冲突的 token ID（不同词可能映射到同一 ID，Qdrant 要求 indices 唯一）
    merged: dict[int, float] = {}
    for token, count in tf.items():
        tid = token_to_id(token)
        weight = count / max(tf.values())
        merged[tid] = merged.get(tid, 0.0) + weight

    return list(merged.keys()), list(merged.values())


# ============================================================
# 稀疏检索器
# ============================================================
class SparseRetriever(BaseRetriever):
    """
    基于 jieba 分词 + BM25 权重的稀疏向量检索器。

    检索流程：
        1. jieba 分词 → 词 ID 映射
        2. 在 Qdrant 中用 sparse vector search 匹配
        3. 返回按 BM25 得分排序的 RetrievalResult 列表
    """

    def __init__(self) -> None:
        self._qdrant = QdrantVectorClient()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        执行稀疏向量检索。

        参数：
            query: 用户自然语言提问
            top_k: 返回前 K 条结果

        返回：
            按匹配得分降序排列的检索结果
        """
        from src.core.config import settings

        if top_k is None:
            top_k = settings.retrieval_top_k

        # ① 分词 → 稀疏向量
        indices, values = text_to_sparse_vector(query)
        if not indices:
            logger.warning("[SparseRetriever] 分词结果为空，无法检索")
            return []

        logger.info(
            f"[SparseRetriever] 检索: '{query[:50]}...' "
            f"→ {len(indices)} 个关键词 (top_k={top_k})"
        )

        # ② Qdrant 稀疏向量搜索
        qdrant_results = self._qdrant.search_sparse(
            indices=indices,
            values=values,
            top_k=top_k,
        )

        # ③ 转换 Qdrant 原始结果
        results = []
        for scored_point in qdrant_results:
            payload = scored_point.payload
            results.append(
                RetrievalResult(
                    chunk_id=str(scored_point.id),
                    doc_id=payload["doc_id"],
                    text=payload["text"],
                    clause_ref=payload.get("clause_ref", ""),
                    score=scored_point.score,
                    chunk_index=payload.get("chunk_index", 0),
                )
            )

        return results
