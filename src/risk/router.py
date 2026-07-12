"""
风险检测 API
============
在 RAG 系统中的角色：
    提供 REST 接口，让用户对已入库的合同发起自动化风险扫描。
    内部调用 A2 关键词引擎 → A3 LLM 二次确认 → 返回最终风险列表。

工作流程：
    POST /api/v1/risk/detect?doc_id=xxx
        │
        ├─ 从 Qdrant 按 doc_id 拉取全部 Chunks
        ├─ rule_engine.scan(chunks) → 关键词粗筛
        ├─ llm_verifier.verify_with_llm(hits) → LLM 精判 + 误报过滤
        └─ 返回 VerifiedRisk 列表 + 统计信息

当前限制（M4 阶段）：
    - 同步返回（非 Celery 异步）。M5 异步化。
    - 仅支持单文档检测。多文档批量扫描列入 V2.0 展望。

用法：
    在 src/api/main.py 中注册：
    from src.risk.router import router
    app.include_router(router)
"""

import json
import logging

from fastapi import APIRouter, Query

from src.core.config import settings
from src.indexing.vector_client import QdrantVectorClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_v1_prefix, tags=["风险检测"])


@router.post("/risk/detect")
async def detect_risks(
    doc_id: str = Query(..., description="已入库合同的文档 ID"),
):
    """
    对指定合同执行风险条款检测。

    流程：
        1. 从 Qdrant 按 doc_id 拉取全部 Chunks
        2. 关键词规则引擎扫描
        3. LLM 对命中的条款逐条二次确认，过滤误报
        4. 返回最终风险列表 + 统计

    返回：
        {
            "doc_id": "xxx",
            "total_chunks": 30,
            "rules_loaded": 13,
            "keyword_hits": 8,
            "verified_risks": 5,
            "false_alarms_filtered": 3,
            "risks": [VerifiedRisk, ...]
        }
    """
    from src.tasks.risk_task import process_risk_detection

    task = process_risk_detection.delay(doc_id=doc_id)
    logger.info(f"[Risk API] 已提交异步任务: task_id={task.id}, doc_id={doc_id[:8]}...")

    return {
        "task_id": task.id,
        "doc_id": doc_id,
        "status": "pending",
        "message": (
            f"风险检测任务已提交（task_id: {task.id}），"
            f"请通过 GET /api/v1/tasks/{task.id} 查询进度。"
            f"完成后可从 result 中获取 session_id。"
        ),
    }


def _get_chunks_by_doc_id(qdrant: QdrantVectorClient, doc_id: str):
    """
    从 Qdrant 按 doc_id 拉取全部 Chunks。

    使用 scroll 按 payload 过滤——一次取出该文档的所有 Point，
    重建为 Chunk 对象（用于 rule_engine 扫描）。
    """
    from src.chunkers.legal_chunker import Chunk

    # scroll 按 filter 拉取全部匹配的 Points
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

    # 重建 Chunk 对象
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

    # 按 chunk_index 恢复原文顺序
    chunks.sort(key=lambda c: c.chunk_index)
    return chunks


def _store_risk_session(
    session_id: str, doc_id: str, risks: list, total_hits: int
) -> None:
    """将风险检测结果 JSON 序列化后存入 Redis，TTL 1 小时"""
    try:
        import redis as rds

        client = rds.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=False,
        )
        key = f"compare:{session_id}"  # 复用 compare key 命名，QA 侧统一读取
        data = json.dumps(
            {
                "type": "risk",
                "doc_id": doc_id,
                "total_risks": len(risks),
                "risks": [
                    {
                        "rule_id": r.rule_id,
                        "rule_name": r.rule_name,
                        "severity": r.severity,
                        "clause_ref": r.clause_ref,
                        "analysis": r.analysis,
                    }
                    for r in risks
                ],
            },
            ensure_ascii=False,
        )
        client.set(key, data, ex=3600)
        logger.info(f"[Risk API] 结果已存入 Redis: compare:{session_id[:8]}...")
    except Exception as e:
        logger.warning(f"[Risk API] Redis 存储失败（不影响检测结果）: {e}")
