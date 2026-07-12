"""
混合检索器（Hybrid Retriever）
==============================
在 RAG 系统中的角色：
    同时走稠密向量检索和稀疏关键词检索两条路，
    用 RRF 算法融合两路排名，输出"兼具语义和字面精度"的最终结果。

RRF 融合算法（学习参考）：
    RRF = Reciprocal Rank Fusion（倒数排名融合）

    公式: score(doc) = Σ (1 / (k + rank_i(doc)))
        k: 平滑参数（通常 60），防止排名为 1 和 2 的权重差距过大
        rank_i(doc): 文档在第 i 路检索中的排名（1-based）

    为什么用 RRF 而非"取最高得分"：
        两条路的得分分布不同——稠密得分在 [0.5, 0.9]，
        稀疏得分在 [0.01, 0.3]。直接比较数字大小会偏袒稠密路。
        RRF 用排名而非绝对得分——排名 1 在两条路上贡献相同。

    示例：
        Dense 排名:  ChunkA=1, ChunkB=2, ChunkC=3
        Sparse 排名: ChunkD=1, ChunkB=2, ChunkA=4

        RRF(ChunkA) = 1/(60+1) + 1/(60+4) = 0.0164 + 0.0156 = 0.0320
        RRF(ChunkB) = 1/(60+2) + 1/(60+2) = 0.0161 + 0.0161 = 0.0322
        RRF(ChunkD) = 1/(60+0) + 1/(60+1) = 1/60   + 0.0164 = 0.0331

        → ChunkD 排第 1（两路之一拿了冠军），ChunkB 排第 2（两路都排第 2），
          正确：B 虽然每路都不在第一，但两路一致认可，综合排名高。

接口保持与 M2 的 BaseRetriever 一致：
    HybridRetriever.retrieve(query, top_k) → List[RetrievalResult]

用法：
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.retrieve("第16条的违约金", top_k=5)
"""

import logging
from typing import List

from src.core.config import settings
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    混合检索器 — 同时执行稠密和稀疏检索，RRF 融合排名。

    组合了两个已有检索器：
        - DenseRetriever: BGE 语义匹配
        - SparseRetriever: jieba 关键词匹配

    不重复实现检索逻辑，只负责融合排名。
    """

    def __init__(self) -> None:
        self._dense = DenseRetriever()
        self._sparse = SparseRetriever()

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        混合检索 + RRF 融合。

        参数：
            query: 用户提问
            top_k: 最终返回的 Top-K 结果

        返回：
            按 RRF 融合得分降序排列的检索结果
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        # 召回多一些（2x），给 RRF 融合更大的候选池
        recall_k = top_k * 2

        logger.info(f"[Hybrid] 双路检索 (dense + sparse, k={recall_k})...")

        # ① 稠密检索
        dense_results = self._dense.retrieve(query, top_k=recall_k)
        # ② 稀疏检索
        sparse_results = self._sparse.retrieve(query, top_k=recall_k)

        # ③ RRF 融合
        merged = self._rrf_fusion(dense_results, sparse_results, k=settings.rrf_k)

        # ④ 截断到 top_k
        final = merged[:top_k]

        for i, r in enumerate(final[:3]):
            logger.info(f"[Hybrid] #{i + 1} rrf={r.score:.4f} [{r.clause_ref}]")

        return final

    # ----------------------------------------------------------
    # RRF 融合算法
    # ----------------------------------------------------------
    def _rrf_fusion(
        self,
        dense: List[RetrievalResult],
        sparse: List[RetrievalResult],
        k: int = 60,
    ) -> List[RetrievalResult]:
        """
        RRF 融合两路检索结果。

        步骤：
            1. 为每个结果分配 RRF 得分：1/(k + rank)
            2. 同一 chunk_id 在两路都有 → RRF 得分累加
            3. 只有一个 chunk_id → 仅该路的 RRF 得分
            4. 按 RRF 得分降序输出

        参数：
            dense: 稠密检索结果（已按 score 降序）
            sparse: 稀疏检索结果（已按 score 降序）
            k: 平滑参数

        返回：
            融合后的结果列表（按 RRF 得分降序）
        """

        # {chunk_id: rrf_score}
        rrf_scores: dict[str, float] = {}
        # {chunk_id: RetrievalResult} — 保留原对象用于最终输出
        seen: dict[str, RetrievalResult] = {}
        # {chunk_id: [rank]} — 用于日志
        rank_info: dict[str, list] = {}

        # ---- 稠密路 RRF 得分 ----
        for rank, doc in enumerate(dense, 1):
            cid = doc.chunk_id
            score = 1.0 / (k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            if cid not in seen:
                seen[cid] = doc
            rank_info.setdefault(cid, []).append(f"d#{rank}")

        # ---- 稀疏路 RRF 得分 ----
        for rank, doc in enumerate(sparse, 1):
            cid = doc.chunk_id
            score = 1.0 / (k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            if cid not in seen:
                seen[cid] = doc
            rank_info.setdefault(cid, []).append(f"s#{rank}")

        # ---- 按 RRF 得分排序 ----
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for cid in sorted_ids:
            doc = seen[cid]
            # 用 RRF 得分替换原始检索得分
            doc.score = round(rrf_scores[cid], 6)
            results.append(doc)

        logger.info(
            f"[Hybrid] RRF 融合: "
            f"dense={len(dense)} + sparse={len(sparse)} "
            f"→ {len(results)} unique (top-3 ranks: "
            f"{[(rank_info.get(r.chunk_id, ['?'])) for r in results[:3]]})"
        )

        return results
