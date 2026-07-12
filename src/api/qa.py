"""
合同问答 API（SSE 流式）
=========================
在 RAG 系统中的角色：
    接收用户的自然语言提问，调用 RAG 图（改写→检索→重排序→生成），
    以 SSE 流式返回 LLM 生成的答案 token，最终附带引用的合同条款来源。

为什么用 SSE 而非同步 JSON：
    - LLM 生成需要 5-10 秒，同步等待体验差
    - SSE 逐 token 推送，用户在检索完成后立即开始看到答案
    - 基于 HTTP，不需要 WebSocket 升级，兼容性好

SSE 事件流格式：
    event: token
    data: {"content": "根"}

    event: token
    data: {"content": "据"}

    event: done
    data: {"session_id": "abc", "sources": [...]}

用法：
    POST /api/v1/qa  {"query": "违约金怎么算？", "session_id": "可选"}

    用 curl 测试：
    curl -X POST http://localhost:8000/api/v1/qa \
      -H "Content-Type: application/json" \
      -d '{"query": "违约金怎么计算？"}' \
      --no-buffer
"""

import json
import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.core.config import settings
from src.domain.models import QARequest
from src.graph.graph_builder import build_rag_graph
from src.utils.redis_client import (
    append_history,
    get_compare_context,
    get_history,
    history_to_text,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_v1_prefix, tags=["问答"])


@router.post("/qa")
async def ask_question(request: QARequest):
    """
    合同问答端点 — SSE 流式返回 LLM 生成的答案。

    事件类型：
        - rewrite: 查询改写结果（可选，调试用）
        - token:   生成的文本片段
        - done:    生成完成，附带 sources 和 session_id
    """
    query = request.query.strip()
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[QA] 收到提问: session={session_id[:8]}... '{query[:80]}'")

    # ---- 读取对话历史 ----
    history = get_history(session_id)
    history_text = history_to_text(history)

    # ---- 读取比对快照（如存在，追加入上下文） ----
    compare_ctx = get_compare_context(session_id)
    if compare_ctx:
        history_text = compare_ctx + "\n\n" + history_text
        logger.info(f"[QA] 关联比对快照: session={session_id[:8]}...")

    # ---- 构建初始状态 ----
    graph = build_rag_graph()
    initial_state = {
        "query": query,
        "session_id": session_id,
        "history": history_text,
        "doc_id": request.doc_id or "",
        "rewritten_query": "",
        "retrieved_docs": [],
        "answer": "",
    }

    # ---- SSE 事件生成器 ----
    async def event_stream():
        full_answer = ""
        final_sources = []
        rewritten_query = ""

        try:
            # astream_events 运行图，自动捕获内部 LLM 的 streaming token
            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event["event"]

                # ---- LLM 流式 token ----
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = getattr(chunk, "content", "")
                    if token:
                        full_answer += token
                        yield (
                            f"event: token\n"
                            f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
                        )

                # ---- 图完成，取最终 state ----
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        full_answer = output.get("answer", full_answer)
                        final_docs = output.get("retrieved_docs", [])
                        rewritten_query = output.get("rewritten_query", "")
                        final_sources = [
                            {
                                "clause_ref": doc.clause_ref,
                                "text_snippet": getattr(doc, "text", "")[:200],
                                "score": round(doc.score, 4),
                            }
                            for doc in final_docs
                        ]

        except Exception as e:
            logger.error(f"[QA] SSE 流异常: {e}")
            yield (
                f"event: error\n"
                f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            )
            return

        # ---- 保存对话到 Redis ----
        if full_answer:
            append_history(
                session_id,
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": full_answer},
                ],
            )

        # ---- 推送最终结果 ----
        yield (
            f"event: done\n"
            f"data: {
                json.dumps(
                    {
                        'session_id': session_id,
                        'rewritten_query': rewritten_query,
                        'sources': final_sources,
                    },
                    ensure_ascii=False,
                )
            }\n\n"
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )
