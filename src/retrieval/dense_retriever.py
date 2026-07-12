"""
稠密向量检索器（Dense Retriever）
===============================
在 RAG 系统中的角色：
    将用户的自然语言提问转为 512 维语义向量，在 Qdrant 中搜索
    余弦距离最近的 K 个条款 Chunk。这是 M2 唯一的检索方式，
    也是 M3 混合检索的"稠密"一极。

检索流程（三步骤）：
    ① 向量化提问: "违约金怎么算？" → BGE 模型 → [0.12, -0.34, ...] (512维)
    ② 相似搜索: 在 Qdrant 的 512 维空间中找最近的 K 个邻居
    ③ 返回结果: 每个结果包含原文 + 条款引用 + 得分

什么是"稠密向量"（dense vector）：
    稠密向量的每个维度都是非零浮点数，由深度学习模型生成。
    它捕获的是"语义相似性"——"违约金"和"违约赔偿"在向量空间中
    距离很近，即使两个词的字面不同。
    对应的"稀疏向量"（M3 添加）基于 BM25 关键词匹配，被搜索词
    更偏"字面匹配"。

为什么嵌入模型需要和索引时一致：
    索引时用 BAAI/bge-small-zh-v1.5 将条款编码为向量，
    检索时必须用同一个模型将提问编码到同一个向量空间。
    用不同的模型（如 text2vec-base-chinese）编码的向量
    虽然也是 512 维，但"语义坐标轴"不同，检索结果会完全错误。

用法：
    from src.retrieval.dense_retriever import DenseRetriever
    retriever = DenseRetriever()
    results = retriever.retrieve("违约责任怎么算？", top_k=5)
    for r in results:
        print(f"[{r.clause_ref}] (score={r.score:.3f}) {r.text[:80]}")
"""

import logging
from typing import List

from src.indexing.embedder import BGEEmbedder
from src.indexing.vector_client import QdrantVectorClient
from src.retrieval.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    稠密向量检索器 — 用 BGE 模型的语义向量做相似度搜索。

    组合了两个 M1 已经建好的模块：
        - BGEEmbedder: 将文本转为 512 维向量
        - QdrantVectorClient: 在向量库中搜索相似向量

    无需额外配置——模型名、Qdrant 连接信息都从 src.core.config 读取。
    """

    def __init__(self) -> None:
        """
        初始化检索器。

        延迟加载说明：
            BGEEmbedder 和 QdrantVectorClient 的 __init__ 不会加载模型或连接数据库。
            BGEEmbedder 的模型加载延迟到第一次 encode() 调用，
            QdrantVectorClient 的连接延迟到第一次 search() 调用。
            所以这里的初始化是零成本的（只创建 Python 对象，不涉及 IO）。
        """
        self._embedder = BGEEmbedder()
        self._qdrant = QdrantVectorClient()

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        检索与用户提问最相关的条款。

        三步流程：
            ① 问句 → 512 维向量（BGEEmbedder）
            ② 向量 → Qdrant 相似搜索（QdrantVectorClient）
            ③ Qdrant 原始结果 → RetrievalResult 对象列表

        参数：
            query: 自然语言提问
            top_k: 返回前 K 条，默认从 settings.RETRIEVAL_TOP_K 读取

        返回：
            按相似度降序排列的结果列表
        """
        # 如果调用方未指定 top_k，使用配置中的默认值
        from src.core.config import settings

        if top_k is None:
            top_k = settings.retrieval_top_k

        logger.info(f"[DenseRetriever] 检索: '{query[:50]}...' (top_k={top_k})")

        # ① 向量化提问
        # encode() 返回 (N, 512) 的 numpy 数组，[0] 取第一条（我们只编码了一个问句）
        query_vector = self._embedder.encode([query])[0]

        # ② 在 Qdrant 中搜索最近的 K 个邻居
        # search() 内部用 Cosine 距离度量——问句向量和每个 Point 的向量做比较
        qdrant_results = self._qdrant.search(query_vector, top_k=top_k)

        # ③ 转换 Qdrant 的原始结果为 RetrievalResult
        retrieval_results = []
        for scored_point in qdrant_results:
            payload = scored_point.payload
            result = RetrievalResult(
                chunk_id=str(scored_point.id),
                doc_id=payload["doc_id"],
                text=payload["text"],
                clause_ref=payload.get("clause_ref", ""),
                score=scored_point.score,
                chunk_index=payload.get("chunk_index", 0),
            )
            retrieval_results.append(result)

        # 日志：打印 top-3 的得分和条款引用（方便调试检索质量）
        for i, r in enumerate(retrieval_results[:3]):
            logger.info(
                f"[DenseRetriever] #{i + 1} score={r.score:.4f} [{r.clause_ref}]"
            )

        return retrieval_results
