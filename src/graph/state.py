"""
LangGraph 状态定义（RAG State）
==============================
在 RAG 系统中的角色：
    定义在检索节点和生成节点之间传递的"数据背包"。
    LangGraph 每执行一个节点，都会从这个 state 中读取输入、
    写入输出，然后传递给下一个节点。

为什么用 TypedDict 而非 Pydantic（学习参考）：
    LangGraph 对 TypedDict 有一等支持——它知道每个字段的类型，
    可以在编译时校验节点之间的数据传递是否正确。
    Pydantic 也可以，但 TypedDict 更轻量（无需额外依赖），
    且与 Python 的类型标注体系天然兼容。

RAGState 的 7 个字段：
    query / session_id / history / doc_id  — 由 API 层写入（history 从 Redis 读出）
    rewritten_query                        — _rewrite_node 产出，供检索使用
    retrieved_docs                         — _retrieve_node 召回，_rerank_node 原地覆盖
    answer                                 — _generate_node 产出，或空结果时由回退分支写入

注意：早期规划中的 confidence 字段最终没有落地。M3 验证发现 cross-encoder
得分是"相对最优排名"而非"绝对相关度"，无法充当门控阈值（详见 graph_builder.py
中 _check_confidence_node 的说明），最终改为在检索预检阶段判空回退。

状态在各个节点之间的流转：
    rewrite → retrieve → rerank → check → generate → END
                 │          │        └ 判 retrieved_docs 是否为空
                 │          └ 原地覆盖 retrieved_docs
                 └ 写入 retrieved_docs（rewrite 阶段写入 rewritten_query）

用法：
    from src.graph.state import RAGState
"""

from typing import List, TypedDict

from src.retrieval.base import RetrievalResult


class RAGState(TypedDict):
    """
    在 LangGraph 节点之间流转的状态对象。

    字段说明：
        query:          用户输入的原始自然语言提问，由 API 层设置，
                        retrieve_node 读取它来检索，generate_node 读取它来组装 prompt。
        retrieved_docs: 检索节点召回的相关合同条款列表，
                        按 score 降序排列。generate_node 从中提取原文喂给 LLM。
        answer:         LLM 生成的最终答案文本，
                        包含对用户问题的回答和对合同条款的引用。
    """

    query: str
    """用户的自然语言提问，例如"违约金怎么计算？" """

    session_id: str
    """会话 ID（UUID），用于 Redis 存取多轮对话历史，供 rewrite_node 消解指代"""

    history: str
    """格式化的对话历史文本（由 API 层从 Redis 读取并写入），供 rewrite_node 使用"""

    doc_id: str
    """可选：限定检索的文档 ID，空字符串表示检索全部合同"""

    rewritten_query: str
    """查询改写后的规范化版本（由 _rewrite_node 生成），用于检索"""

    retrieved_docs: List[RetrievalResult]
    """检索召回的合同条款列表，按相似度降序排列"""

    answer: str
    """LLM 生成的答案文本，包含条款引用"""
