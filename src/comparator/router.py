"""
合同比对 API
============
在 RAG 系统中的角色：
    提供 REST 接口，对两份已入库合同进行自动化条款级比对。

端点的核心流程：
    POST /api/v1/compare  {"doc_ids": ["id_a", "id_b"]}
        │
        ├─ 校验 doc_ids 数量（当前仅支持 N=2）
        ├─ 从 Qdrant 分别拉取两份合同的全部 Chunks
        ├─ diff_analyzer.compare_documents() → B0→B1→B2→B3
        ├─ 结果 JSON 序列化后存入 Redis compare:{session_id}
        └─ 返回 CompareResult + session_id（用于后续追问）

多文档支持说明：
    API 签名设计为 list[doc_id]，为 N≥2 留好扩展空间。
    当前实现仅支持 len(doc_ids) == 2，传入其他数量返回 400。

用法：
    在 src/api/main.py 中注册：
    from src.comparator.router import router
    app.include_router(router)
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_v1_prefix, tags=["合同比对"])


# ============================================================
# 请求/响应模型
# ============================================================
class CompareRequest(BaseModel):
    """比对请求"""

    doc_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="要比对的文档 ID 列表，当前仅支持 2 份",
        examples=[["abc-123", "def-456"]],
    )


class CompareResponse(BaseModel):
    """比对响应"""

    session_id: str = Field(
        ..., description="比对会话 ID，用于后续追问（传入 QA 端点）"
    )
    doc_id_a: str
    doc_id_b: str
    relevance_score: float = Field(
        ..., description="两份合同全文的语义相似度（B0 预检结果）"
    )
    stats: dict = Field(
        ...,
        description="比对统计：total_aligned/identical/with_differences/only_in_a/only_in_b",
    )
    differences: list[dict] = Field(
        default_factory=list,
        description="差异项列表（含 LLM 分析）",
    )
    only_in_a: list[dict] = Field(
        default_factory=list,
        description="仅 A 合同有的条款（clause_ref + text 片段）",
    )
    only_in_b: list[dict] = Field(
        default_factory=list,
        description="仅 B 合同有的条款（clause_ref + text 片段）",
    )


# ============================================================
# API 端点
# ============================================================
@router.post("/compare")
async def compare_contracts(request: CompareRequest):
    """
    对两份已入库合同进行条款级比对。

    请求示例：
        POST /api/v1/compare
        {"doc_ids": ["abc-123", "def-456"]}

    处理流程：
        1. 校验 doc_ids 数量（当前仅支持 2）
        2. 从 Qdrant 拉取两份合同的全部 Chunks
        3. B0 全文相关性预检 → B1 条款对齐 → B2 预过滤 → B3 LLM 对比
        4. 结果存入 Redis compare session（TTL 1 小时），支持后续追问

    返回：
        比对统计 + 差异列表 + session_id（用于 QA 追问"
        例如在 QA 请求中传入同一 session_id，系统会感知比对上下文）
    """
    doc_ids = request.doc_ids

    if len(doc_ids) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"当前仅支持比对 2 份合同，收到 {len(doc_ids)} 份。多文档比对列入 V2.0 规划。",
        )

    from src.tasks.compare_task import process_comparison

    doc_id_a, doc_id_b = doc_ids[0], doc_ids[1]
    task = process_comparison.delay(doc_id_a=doc_id_a, doc_id_b=doc_id_b)
    logger.info(
        f"[Compare] 已提交异步任务: task_id={task.id}, "
        f"{doc_id_a[:8]}... vs {doc_id_b[:8]}..."
    )

    return {
        "task_id": task.id,
        "doc_id_a": doc_id_a,
        "doc_id_b": doc_id_b,
        "status": "pending",
        "message": (
            f"合同比对任务已提交（task_id: {task.id}），"
            f"请通过 GET /api/v1/tasks/{task.id} 查询进度。"
            f"完成后可从 result 中获取 session_id。"
        ),
    }


# ============================================================
# 辅助函数
# ============================================================
def _get_chunks_by_doc_id(qdrant, doc_id: str):
    """从 Qdrant 按 doc_id 拉取全部 Chunks，按 chunk_index 排序"""
    from src.chunkers.legal_chunker import Chunk
    from qdrant_client.http import models as qmodels

    all_points = []
    offset = None

    while True:
        points, next_offset = qdrant._client.scroll(
            collection_name=qdrant._collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    chunks = []
    for pt in all_points:
        payload = pt.payload
        chunks.append(
            Chunk(
                chunk_id=str(pt.id),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", 0),
                clause_ref=payload.get("clause_ref", ""),
                start_char=0,
                end_char=len(payload.get("text", "")),
            )
        )

    chunks.sort(key=lambda c: c.chunk_index)
    return chunks


def _store_compare_session(session_id: str, result) -> None:
    """将比对结果 JSON 序列化后存入 Redis，TTL 1 小时"""
    try:
        import redis as rds
        from src.core.config import settings as s

        client = rds.Redis(
            host=s.redis_host,
            port=s.redis_port,
            db=s.redis_db,
            decode_responses=False,
        )
        key = f"compare:{session_id}"
        data = json.dumps(
            {
                "doc_id_a": result.doc_id_a,
                "doc_id_b": result.doc_id_b,
                "stats": {
                    "total_aligned": result.stats.total_aligned,
                    "identical": result.stats.identical,
                    "with_differences": result.stats.with_differences,
                    "only_in_a": result.stats.only_in_a,
                    "only_in_b": result.stats.only_in_b,
                },
            },
            ensure_ascii=False,
        )
        client.set(key, data, ex=3600)
        logger.info(f"[Compare] 结果已存入 Redis: compare:{session_id[:8]}...")
    except Exception as e:
        logger.warning(f"[Compare] Redis 存储失败（不影响比对结果）: {e}")
