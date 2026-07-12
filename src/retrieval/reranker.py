"""
Cross-Encoder 重排序器（Reranker）
==================================
在 RAG 系统中的角色：
    对混合检索召回的候选条款做精排——cross-encoder 同时"看到"query 和 chunk，
    输出精准的相关性得分，将最匹配的条款排到最前面喂给 LLM。

Bi-encoder vs Cross-encoder（学习参考）：
    ┌──────────────┬─────────────────────┬──────────────────────┐
    │              │ Bi-encoder (BGE)     │ Cross-encoder         │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ 编码方式     │ query/chunk 分别编码  │ query+chunk 联合编码   │
    │ 速度         │ 快（chunk 预编码存库）│ 慢（每个 pair 现场算）  │
    │ 精度         │ 中等（间接相似度）    │ 高（直接判断相关性）    │
    │ 适用阶段     │ 粗筛（N → Top-K）     │ 精排（Top-K → Top-N）  │
    └──────────────┴─────────────────────┴──────────────────────┘

    举例说明：
        query = "违约金怎么算？"
        chunk1 = "第十六条 旅行社的违约责任..."（强相关）
        chunk2 = "第二十一条 旅游费用及支付..."（弱相关）

        BGE 可能给两者接近的分数（都涉及"费用"），但 cross-encoder
        在读了 query+chunk 后能准确判断：chunk1 得分 0.9，chunk2 得分 0.3。

为什么放在重排序而非替代 BGE：
    直接把 cross-encoder 用于全量检索会非常慢——30 个 chunk × 每个 pair 都要
    过模型 = O(N)。先用 BGE 召回 20 条，再精排 20 条 = O(20)，
    速度可接受、精度也高。这是标准 RAG 的"召回→精排"两阶段架构。

使用的模型：
    cross-encoder/ms-marco-MiniLM-L-6-v2
    - 384 维，~200MB
    - 微调自 MS MARCO（微软问答数据集），对 QA 场景优化
    - 虽然是英文模型，但对中文 cross-encoding 仍有很好的迁移效果
      （因为判断相关性靠的是语义结构而非语言本身）

统一约定（从 embedder.py 的教训中吸收）：
    - local_files_only=True 优先，不向 HF Hub 发起在线检查请求
    - HF_HUB_OFFLINE=1 作为双保险
    - 本地无模型时报错提示国内镜像，绝不因网络超时卡住

用法：
    from src.retrieval.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("违约金怎么算？", docs)
"""

import logging
import os
import time
from typing import List

from src.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)

# ---- HF 离线策略（与 embedder.py 一致）----
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class CrossEncoderReranker:
    """
    使用 cross-encoder 对检索结果做精排。

    模型延迟加载，首次 rerank() 调用时才加载。
    """

    def __init__(self, model_name: str | None = None) -> None:
        from src.core.config import settings

        self._model_name = model_name or settings.cross_encoder_model
        self._model = None

    @property
    def model(self):
        """延迟加载 cross-encoder 模型（与 BGEEmbedder 相同的策略）"""
        if self._model is None:
            logger.info(f"[Reranker] 正在加载模型: {self._model_name}...")
            load_start = time.perf_counter()

            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers 未安装。请运行: uv add sentence-transformers"
                )

            # ---- 加载策略（与 embedder.py 一致）----
            # 优先本地，本地没有时才联网下载
            try:
                logger.info("[Reranker] 尝试从本地缓存加载（不联网）...")
                self._model = CrossEncoder(
                    self._model_name,
                    local_files_only=True,
                )
                logger.info("[Reranker] 模型从本地缓存加载成功")
            except Exception:
                logger.info("[Reranker] 本地缓存未找到模型，尝试从 HuggingFace 下载...")
                try:
                    self._model = CrossEncoder(
                        self._model_name,
                        local_files_only=False,
                    )
                except OSError as e:
                    error_msg = str(e).lower()
                    if any(
                        kw in error_msg
                        for kw in [
                            "connection",
                            "timeout",
                            "hub",
                            "huggingface",
                            "resolve",
                            "refused",
                        ]
                    ):
                        raise RuntimeError(
                            f"无法加载 Cross-Encoder 模型 "
                            f"'{self._model_name}'。\n"
                            f"\n"
                            f"模型既不在本地缓存中，也无法从 HuggingFace 下载。\n"
                            f"\n"
                            f"解决方案：\n"
                            f"  1. 一次性下载 → 在网络可用的条件下运行一次\n"
                            f"  2. 国内镜像 → 设置环境变量 "
                            f"HF_ENDPOINT=https://hf-mirror.com\n"
                            f"  3. 手动下载 → "
                            f"将模型文件放入 '{self._model_name}' 对应目录\n"
                            f"\n"
                            f"原始错误: {e}"
                        ) from e
                    raise

            load_time = (time.perf_counter() - load_start) * 1000
            logger.info(f"[Reranker] 模型加载完成，耗时 {load_time:.0f}ms")

        return self._model

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------
    def rerank(
        self,
        query: str,
        docs: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        对检索结果做精排。

        参数：
            query: 用户原始提问
            docs:  检索召回的候选条款列表（通常 10-20 条）

        返回：
            重新排序后的结果列表（数量不变，但顺序按新得分排列）

        工作原理：
            CrossEncoder.predict([(query, doc1.text), (query, doc2.text), ...])
            → [score1, score2, ...]
            → 用新得分替换原 RRF 得分
            → 按新得分降序排列

        注意：
            - doc.score 会被 CrossEncoder 的新得分覆盖
            - 模型加载在首次调用时发生（约 2-5 秒）
            - 推理耗时 ~100-300ms × len(docs)（取决于 CPU）
        """
        if not docs:
            return []

        logger.info(
            f"[Reranker] 开始精排 {len(docs)} 条候选 (query: '{query[:50]}...')"
        )
        start = time.perf_counter()

        # ① 构建 (query, chunk_text) pair 列表
        pairs = [(query, doc.text) for doc in docs]

        # ② 批量推理 — CrossEncoder 一次性对所有 pair 打分
        scores = self.model.predict(pairs, show_progress_bar=False)

        # predict 返回的可能是 numpy array 或 list，统一处理
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        # ③ 用新得分替换原得分
        for doc, new_score in zip(docs, scores):
            doc.score = float(new_score)

        # ④ 按新得分降序排列（得分越高越相关）
        docs_sorted = sorted(docs, key=lambda d: d.score, reverse=True)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"[Reranker] 精排完成: {len(docs)} 条, "
            f"{elapsed:.0f}ms ({elapsed / len(docs):.0f}ms/条)"
        )
        if docs_sorted:
            logger.info(
                f"[Reranker] Top-1: score={docs_sorted[0].score:.4f} "
                f"[{docs_sorted[0].clause_ref}]"
            )

        return docs_sorted
