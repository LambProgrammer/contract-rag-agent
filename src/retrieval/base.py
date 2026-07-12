"""
检索层抽象基类
=============
在 RAG 系统中的角色：
    定义检索的统一接口。所有检索策略（稠密向量、稀疏关键词、混合）
    都实现同一个 BaseRetriever，上层调用方不需要关心底层用了哪种检索方式。

为什么需要抽象基类（学习参考）：
    M2 只需要稠密向量检索（DenseRetriever），但 M3 要加：
        - SparseRetriever（BM25 关键词匹配）
        - HybridRetriever（稠密 + 稀疏融合）
    如果现在把"问句向量化 → Qdrant search"直接写在 QA 逻辑里，
    M3 就要改核心代码。有 BaseRetriever 之后：
        M2: DenseRetriever  ← 实现接口
        M3: SparseRetriever ← 新增实现，不改旧代码
        M3: HybridRetriever ← 组合前两者，不改旧代码

RetrievalResult vs Chunk（学习参考）：
    为什么不用 M1 已有的 Chunk 类作为检索结果？
        - Chunk 是分块阶段的数据模型（关注 start_char、end_char 等位置信息）
        - RetrievalResult 是检索阶段的数据模型（关注 score 等排名信息）
        - 职责分离：两个阶段用不同的模型，互不污染

用法：
    from src.retrieval.base import BaseRetriever, RetrievalResult
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class RetrievalResult:
    """
    一次检索返回的单个结果。

    与 M1 的 Chunk 类的区别：
        - Chunk: 分块时创建，字段偏"文档元数据"（start_char, end_char）
        - RetrievalResult: 检索时创建，字段偏"匹配度"（score）

    字段说明：
        - score: Cosine 相似度，范围通常 0-1（归一化后）。
          0.9+ = 高度相关（条款直接回答了问题）
          0.7-0.9 = 相关（条款内容与问题有关）
          0.5-0.7 = 弱相关（可能只涉及部分关键词）
          <0.5 = 不太相关（M2 阶段建议过滤）
    """

    chunk_id: str
    """Chunk 唯一 ID，用于追溯"""

    doc_id: str
    """来源文档 ID"""

    text: str
    """条款原文——检索到的最核心产出，直接喂给 LLM"""

    clause_ref: str
    """条款引用标识（如"第十六条 违约责任"），用于 LLM 答案中的出处标注"""

    score: float
    """Cosine 相似度得分，越高越相关"""

    chunk_index: int
    """块在原文中的序号，保持原文顺序"""


class BaseRetriever(ABC):
    """
    检索器的抽象基类。

    所有检索实现必须提供 retrieve() 方法。
    输入是自然语言提问，输出是排序后的条款结果列表。

    接口设计原则：
        - 只暴露必要参数（query, top_k），不过早扩展
        - 返回排序好的列表（按 score 降序），调用方不需要自己排序
    """

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, doc_id: str = ""
    ) -> List[RetrievalResult]:
        """
        根据用户提问检索最相关的条款。

        参数：
            query: 用户的自然语言提问（如"违约金怎么计算？"）
            top_k: 返回最相关的前 K 条结果，默认 5

        返回：
            RetrievalResult 列表，按 score 降序排列（最相关的排在最前面）
        """
        ...
