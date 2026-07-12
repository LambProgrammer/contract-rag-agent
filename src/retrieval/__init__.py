"""
检索模块（Retrieval）
====================
在 RAG 系统中的角色：
    将用户提问转为向量，在 Qdrant 中搜索最相关的合同条款。
    这是 RAG 中 "R"（Retrieval）的核心实现。

模块结构：
    base.py              — 抽象基类 BaseRetriever + RetrievalResult
    dense_retriever.py   — 稠密向量检索（M2）
    sparse_retriever.py  — 稀疏向量检索（M3 新增，jieba + BM25）
    hybrid_retriever.py  — 混合检索（M3 新增，Dense + Sparse + RRF 融合）
    reranker.py          — Cross-Encoder 精排（M3 新增）

用法（推荐）：
    from src.retrieval import HybridRetriever  # M3 混合检索
    retriever = HybridRetriever()
    results = retriever.retrieve("第16条规定的违约金？")
"""

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse_retriever import SparseRetriever

__all__ = [
    "BaseRetriever",
    "CrossEncoderReranker",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "SparseRetriever",
]
