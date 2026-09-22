"""
LangGraph 图构建器
==================
在 RAG 系统中的角色：
    将"检索"和"LLM 生成"两个节点编排为一条可控的执行流。
    这是 RAG 中 "A"（Augmented）的核心编排层——检索到的条款
    作为上下文"增强"了 LLM 的生成能力。

为什么用 LangGraph 而非直接函数调用（学习参考）：
    三个理由，对应 M3 的三个扩展点：

    1. 可插拔节点：M3 加查询改写节点，只需 add_node + 改边，不改现有代码
       原为 retrieve → generate，加入 rewrite 后变为 rewrite → retrieve → generate

    2. 条件分支：M3 加"检索结果为空"的判断，为空则跳过生成、直接走回退
       if not retrieved_docs: goto END  else: goto generate
       （早期曾规划用 cross-encoder 置信度阈值做门控，M3 验证证明其不成立，
         详见 _check_confidence_node 的说明）

    3. 可观测性：每个节点的输入/输出都被 LangGraph 自动记录，
       结合 LangFuse（M5）可以可视化追踪每次检索和生成的质量

M2 的图结构（最简单的线性流程）：
    START → retrieve_node → generate_node → END

M3 的最终图结构：
    START → rewrite → retrieve → rerank → check ─[docs 非空]→ generate → END
                                                └[docs 为空]→ END（回退文本已写入 answer）

用法：
    from src.graph.graph_builder import build_rag_graph
    graph = build_rag_graph()
    result = graph.invoke({"query": "违约责任怎么算？"})
    print(result["answer"])
"""

import logging

from langfuse import observe
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr

from src.core.config import settings
from src.graph.state import RAGState

logger = logging.getLogger(__name__)


# ============================================================
# 节点 0：查询改写（rewrite_node）— M3 新增
# ============================================================
@observe(name="rewrite-node", as_type="chain")
def _rewrite_node(state: RAGState) -> dict:
    """
    查询改写节点 — 将口语化提问规范化为适合检索的标准表述，
    并利用对话历史消解指代。

    数据流：
        输入: state["query"]          — 用户原始提问
              state["history"]        — 对话历史文本（由 API 层从 Redis 读取）
        输出: rewritten_query          — 改写后的规范化查询

    改写能力：
        1. 口语 → 书面语（"不干了赔多少"→"解除合同的违约金计算方式"）
        2. 指代消解 — 利用 history 确定"这个条款"/"那"指什么
           （需要 history 中有上下文，无历史则跳过）
        3. 纠正明显错别字
        4. 已规范的提问直接照原样返回
    """
    from langchain_deepseek import ChatDeepSeek

    raw_query = state["query"]
    history = state.get("history", "")
    logger.info(f"[Graph:rewrite] 原始提问: '{raw_query[:80]}'")

    llm = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=SecretStr(settings.deepseek_api_key),
        temperature=0.1,
    )

    # 对话历史作为消解指代的上下文（没有则跳过）
    history_block = ""
    if history and history != "（无对话历史）":
        history_block = (
            f"对话历史：\n{history}\n\n"
            f"如果当前提问中包含指代词（如'这个条款'、'那'、'它'），"
            f"请根据对话历史确定指代的具体内容并融入改写后的查询；"
            f"如果无法确定指代，保持原样。\n\n"
        )

    rewrite_prompt = (
        "你是一个合同查询改写助手。将用户的原始提问改写为适合检索合同条款的规范查询。\n\n"
        f"{history_block}"
        "规则：\n"
        "1. 将口语化表述转为书面法律术语"
        "（例如'不干了赔多少'→'解除合同的违约金计算方式'）\n"
        "2. 纠正明显的错别字和拼写错误\n"
        "3. 如果原始提问已经很规范，直接返回原问题，不要做无意义的改写\n\n"
        f"原始提问：{raw_query}\n\n"
        "改写后："
    )

    response = llm.invoke(rewrite_prompt)
    content = response.content
    # LangChain content 类型标注为 str | list（兼容多模态），DeepSeek 实际返回 str
    rewritten = content.strip() if isinstance(content, str) else str(content)

    if not rewritten:
        rewritten = raw_query

    logger.info(f"[Graph:rewrite] 改写: '{rewritten[:80]}'")
    return {"rewritten_query": rewritten}


# ============================================================
# 节点 1：检索（retrieve_node）
# ============================================================
@observe(name="retrieve-node", as_type="retriever")
def _retrieve_node(state: RAGState) -> dict:
    """
    检索节点 — 将用户提问转为向量，在 Qdrant 中搜索最相关条款。

    数据流：
        输入: state["query"]     — 用户原始提问
        输出: retrieved_docs     — 召回的相关条款列表

    为什么在节点内部创建 DenseRetriever（而非外部注入）：
        LangGraph 节点函数是无状态的——每次调用都可能在不同的
        线程或进程中执行。在节点内部创建实例确保了线程安全。
    """
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.sparse_retriever import SparseRetriever

    query = state.get("rewritten_query", "") or state["query"]
    logger.info(f"[Graph:retrieve] 检索: '{query[:60]}'")

    # ============================================================
    # 预检：两路各取 Top-1，用绝对得分判断是否与合同相关
    #
    # Cosine 得分反映语义距离：>0.5 高度相关，0.3-0.5 弱相关，<0.3 基本无关
    # BM25 绝对零：问句中没有任何词命中合同中的词 → 字面完全不相关
    #
    # 两路同时不达标 → 合同中没有相关内容 → 直接返回空，触发 fallback
    # ============================================================
    dense = DenseRetriever()
    sparse = SparseRetriever()
    dense_top1 = dense.retrieve(query, top_k=1)
    sparse_top1 = sparse.retrieve(query, top_k=1)

    dense_dead = not dense_top1 or dense_top1[0].score < 0.30
    # 稀疏得分为 0.0 意味着用户提问中的词在合同原文里一个都找不到
    # < 0.001 是为了规避浮点数零值误判，实际等价于"完全无匹配"
    sparse_dead = not sparse_top1 or sparse_top1[0].score < 0.001

    if dense_dead and sparse_dead:
        logger.warning(
            f"[Graph:retrieve] 双路均无有效匹配 → 返回空，触发 fallback "
            f"(dense_top1={'无' if not dense_top1 else f'{dense_top1[0].score:.3f}'}, "
            f"sparse_top1={'无' if not sparse_top1 else '有'})"
        )
        return {"retrieved_docs": []}

    # ---- 预检通过 → 正常混合检索 + 重排序 ----
    retriever = HybridRetriever()
    docs = retriever.retrieve(query, top_k=settings.retrieval_top_k)

    # 按 doc_id 过滤（如果用户指定了合同范围）
    target_doc = state.get("doc_id", "")
    if target_doc:
        docs = [d for d in docs if d.doc_id == target_doc]
        logger.info(
            f"[Graph:retrieve] doc_id 过滤后: {len(docs)} 条 (target: {target_doc[:8]}...)"
        )

    logger.info(f"[Graph:retrieve] 混合检索完成: {len(docs)} 条候选")
    return {"retrieved_docs": docs}


# ============================================================
# 节点 2：重排序（rerank_node）— M3 新增
# ============================================================
@observe(name="rerank-node", as_type="chain")
def _rerank_node(state: RAGState) -> dict:
    """
    重排序节点 — 用 CrossEncoder 对检索召回结果精排。

    数据流：
        输入: state["query"]           — 用户原始提问
              state["retrieved_docs"]  — 粗筛候选（~20 条）
        输出: retrieved_docs           — 精排后（~5 条）

    为什么需要"重排序"：
        混合检索的 RRF 融合只用了"排名位置"信息，不知道 query 和 chunk
        之间的细粒度语义关系。CrossEncoder 直接读入 (query, chunk) pair
        输出精准得分，将最相关的条款排到 LLM 的前面。

        LLM 对 prompt 中靠前的文字更关注——精排后前几条越准确，
        LLM 生成的答案质量越高。
    """
    from src.retrieval.reranker import CrossEncoderReranker

    docs = state["retrieved_docs"]

    if not docs:
        logger.warning("[Graph:rerank] 无候选结果，跳过重排序")
        return {"retrieved_docs": docs}

    logger.info(f"[Graph:rerank] 开始 CrossEncoder 精排 ({len(docs)} 条)...")

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(state["query"], docs)[: settings.rerank_top_k]

    logger.info(f"[Graph:rerank] 精排完成: {len(reranked)} 条 → LLM")
    return {"retrieved_docs": reranked}


# ============================================================
# 节点 3：门控检查（check_confidence_node）— M3 新增
# ============================================================
@observe(name="check-node", as_type="chain")
def _check_confidence_node(state: RAGState) -> dict:
    """
    门控检查节点 — 唯一职责：docs 为空时直接返回 fallback。

    为什么不做 cross-encoder 阈值检查：
        cross-encoder 的得分是"相对最优"而非"绝对相关"——只要候选池
        不为空，它就会给出高分。用 cross-encoder 得分做门控相当于
        "在一堆不相关的东西里选最像的，然后问最像的够不够像"，
        cross-encoder 永远会说"够像"。这是一个数学上的错误用法。

    真正的门控已经由 _retrieve_node 的检索预检完成：
        Dense Cosine < 0.3 且 Sparse 为零关键词 → 直接返回 []，
        跳过 rerank 和 generate，到这里触发 fallback。

    弱相关情况（检索预检通过但条款无法回答用户问题）：
        由 generate_node 中的 LLM 自行判断——LLM 读条款后如果发现
        答不了，会诚实告知用户。LLM 的判断比任何数值阈值都可靠。
    """
    docs = state.get("retrieved_docs", [])

    if not docs:
        logger.warning("[Graph:check] 无检索结果，返回 fallback")
        return {
            "answer": (
                "抱歉，在已上传的合同中未找到与您问题相关的条款。\n\n"
                "建议：\n"
                "1. 确认合同是否已上传并解析完成\n"
                "2. 尝试使用不同的表述方式重新提问\n"
                "3. 检查提问中涉及的条款编号是否存在于合同中"
            )
        }

    logger.info("[Graph:check] 检索结果非空，放行到 generate_node")
    return {}


def _should_fallback(state: RAGState) -> str:
    """条件路由：answer 已有值（即 docs 为空时已 preset）→ 结束"""
    if state.get("answer", ""):
        return "end"
    return "generate"


# ============================================================
# 节点 4：生成（generate_node）
# ============================================================
@observe(name="generate-node", as_type="generation")
def _generate_node(state: RAGState) -> dict:
    """
    生成节点 — 将检索到的条款原文 + 用户问题组装为 prompt，调用 LLM 生成答案。

    数据流：
        输入: state["retrieved_docs"]  — 检索召回的相关条款
              state["query"]           — 用户原始提问
        输出: answer                   — LLM 生成的答案文本

    Prompt 设计（学习参考）：
        RAG 的 prompt 分为三个逻辑部分：
        1. 系统指令：告诉 LLM 它的角色和行为规则
        2. 上下文：检索到的合同条款原文（这是 RAG 的核心——"增强"检索结果）
        3. 用户问题：用户原始提问

        这三部分缺一不可：
        - 缺系统指令 → LLM 可能编造条款内容
        - 缺上下文 → 就是普通的 LLM 问答，不是 RAG
        - 缺用户问题 → LLM 不知道要回答什么
    """
    from langchain_deepseek import ChatDeepSeek

    docs = state["retrieved_docs"]
    query = state["query"]

    # ---- 组装检索到的条款原文作为上下文 ----
    if docs:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"【条款{i}】来源：{doc.clause_ref}\n{doc.text}")
        context = "\n\n".join(context_parts)
    else:
        context = "（未找到相关合同条款）"

    # ---- 提取历史中的风险/比对上下文 ----
    supplementary = ""
    history = state.get("history", "")
    if history and ("[风险检测结果]" in history or "[比对上下文]" in history):
        supplementary = history + "\n\n"
        logger.info("[Graph:generate] 检测到风险/比对上下文，已并入 prompt")

    # ---- 组装 prompt ----
    prompt = (
        f"{supplementary}"
        "你是一个专业的合同审查助手。请根据以下合同条款原文，"
        "回答用户关于合同内容的问题。\n\n"
        "规则（请严格遵循）：\n"
        "1. 答案基于合同条款原文和对话历史中的风险检测/合同比对结果，"
        "两者地位平等。风险/比对分析中已经给出的任何分析结论和修改建议，"
        "都是你作为审查助手产出的专业判断，应当直接采纳并告知用户，不要重复审问\n"
        "2. 不得编造或猜测\n"
        "3. 引用条款时请注明出处（如《根据第十六条》）\n"
        "4. 先判断检索到的条款是否与用户问题真正相关、是否足以回答问题：\n"
        "   - 如果条款能够回答用户的问题 → 基于条款给出完整答案\n"
        "   - 如果条款包含相关关键词但内容实际上与问题无关（弱相关）\n"
        "     → 明确告知用户'检索到的合同条款与您的问题不完全匹配'，"
        "     说明条款涉及的内容与用户问题的差异，并建议用户换个角度提问\n"
        "   - 如果条款完全无法回答用户的问题 → 直接告知用户，"
        "     说明合同中未找到相关内容，并提供具体的提问建议\n"
        "5. 使用专业但易懂的中文回答\n\n"
        f"=== 合同条款原文 ===\n"
        f"{context}\n\n"
        f"=== 用户问题 ===\n"
        f"{query}\n\n"
        f"请回答："
    )

    logger.info(f"[Graph:generate] 调用 LLM ({settings.deepseek_model})...")

    # ---- 调用 LLM ----
    # ChatDeepSeek 是 langchain-deepseek 提供的 LangChain 兼容封装
    llm = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=SecretStr(settings.deepseek_api_key),
        temperature=0.3,
        streaming=True,  # 启用流式 — 配合 astream_events 逐 token 推送
    )
    response = llm.invoke(prompt)
    answer = response.content

    logger.info(f"[Graph:generate] LLM 回答完成: {len(answer)} 字符")
    return {"answer": answer}


# ============================================================
# 图构建
# ============================================================
def build_rag_graph():
    """
    构建 RAG 工作流图。

    返回：
        编译后的 LangGraph 可执行图，调用 graph.invoke({"query": "..."}) 即可运行。

    M2 图结构（线性）：
        START → retrieve → generate → END

    M3 step1 图结构（混合检索）：
        START → retrieve (hybrid) → generate → END

    M3 step2 图结构（混合检索 + 重排序）：
        START → retrieve (20条) → rerank (5条) → generate → END

    M3 step3 图结构（改写 + 混合检索 + 重排序 + 置信度回退）：
        START → rewrite → retrieve (20条) → rerank (5条)
                → check → [高置信度] → generate → END
                        → [低置信度] → END (fallback)
    """
    graph = StateGraph(RAGState)

    # ---- 注册全部节点 ----
    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("rerank", _rerank_node)
    graph.add_node("check", _check_confidence_node)
    graph.add_node("generate", _generate_node)

    # ---- 构建边 ----
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check")

    # 条件边：check 之后根据置信度决定走 generate 还是直接结束
    graph.add_conditional_edges(
        "check",
        _should_fallback,
        {
            "generate": "generate",
            "end": END,
        },
    )

    graph.add_edge("generate", END)

    logger.info(
        "[Graph] RAG 图编译完成 "
        "(rewrite → retrieve → rerank → check → generate/fallback)"
    )

    return graph.compile()
