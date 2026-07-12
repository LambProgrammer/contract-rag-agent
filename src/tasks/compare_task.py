"""
合同比对异步任务（process_comparison）
======================================
在 RAG 系统中的角色：
    将 M4 的同步比对端点改造为 Celery 异步任务。
    解决长耗时比对（大量 LLM 调用）导致 HTTP 网关超时的问题。

与 process_contract 相同的异步模式：
    API 端 → task.delay() → 秒返 task_id → Worker 后台执行 → 前端轮询进度

用法：
    from src.tasks.compare_task import process_comparison
    task = process_comparison.delay(doc_id_a="xxx", doc_id_b="yyy")
"""

import json
import logging
import uuid

from celery import shared_task

from src.core.config import settings

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=1, name="process_comparison")
def process_comparison(self, doc_id_a: str, doc_id_b: str) -> dict:
    """
    异步执行跨合同比对。

    参数：
        doc_id_a: 合同 A 的文档 ID
        doc_id_b: 合同 B 的文档 ID

    返回：
        {"doc_id_a": ..., "doc_id_b": ..., "session_id": ..., "stats": ..., "differences": [...]}
    """
    self.update_state(state="PROCESSING", meta={"progress": 5, "stage": "比对"})

    logger.info(f"[CompareTask] 开始比对: {doc_id_a[:8]}... vs {doc_id_b[:8]}...")

    # ---- 从 Qdrant 拉取两份合同的 Chunks ----
    from src.indexing.vector_client import QdrantVectorClient

    qdrant = QdrantVectorClient()
    from src.risk.router import _get_chunks_by_doc_id

    chunks_a = _get_chunks_by_doc_id(qdrant, doc_id_a)
    chunks_b = _get_chunks_by_doc_id(qdrant, doc_id_b)
    if not chunks_a:
        raise ValueError(f"未找到文档 {doc_id_a} 的索引数据")
    if not chunks_b:
        raise ValueError(f"未找到文档 {doc_id_b} 的索引数据")

    self.update_state(
        state="PROCESSING",
        meta={"progress": 20, "stage": "条款对齐"},
    )

    # ---- 执行比对 ----
    from src.comparator.diff_analyzer import compare_documents

    result = compare_documents(doc_id_a, doc_id_b, chunks_a, chunks_b)

    # ---- 生成 session_id 并存入 Redis ----
    session_id = str(uuid.uuid4())
    try:
        import redis as rds

        client = rds.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=False,
        )
        diff_summaries = []
        for d in result.differences:
            if not d.is_identical:
                diff_summaries.append(
                    {
                        "clause_ref_a": d.clause_ref_a,
                        "clause_ref_b": d.clause_ref_b,
                        "analysis": d.llm_analysis,
                    }
                )

        data = json.dumps(
            {
                "doc_id_a": doc_id_a,
                "doc_id_b": doc_id_b,
                "stats": {
                    "total_aligned": result.stats.total_aligned,
                    "identical": result.stats.identical,
                    "with_differences": result.stats.with_differences,
                    "only_in_a": result.stats.only_in_a,
                    "only_in_b": result.stats.only_in_b,
                },
                "differences": diff_summaries,
            },
            ensure_ascii=False,
        )
        client.set(f"compare:{session_id}", data, ex=3600)
    except Exception as e:
        logger.warning(f"[CompareTask] Redis 存储失败: {e}")

    output = {
        "doc_id_a": doc_id_a,
        "doc_id_b": doc_id_b,
        "session_id": session_id,
        "relevance_score": result.relevance_score,
        "stats": {
            "total_aligned": result.stats.total_aligned,
            "identical": result.stats.identical,
            "with_differences": result.stats.with_differences,
            "only_in_a": result.stats.only_in_a,
            "only_in_b": result.stats.only_in_b,
        },
        "differences": [
            {
                "clause_ref_a": d.clause_ref_a,
                "clause_ref_b": d.clause_ref_b,
                "align_method": d.align_method,
                "is_identical": d.is_identical,
                "llm_analysis": d.llm_analysis,
            }
            for d in result.differences
        ],
        "only_in_a": [
            {"clause_ref": c.clause_ref, "text_snippet": c.text[:200]}
            for c in result.only_in_a
        ],
        "only_in_b": [
            {"clause_ref": c.clause_ref, "text_snippet": c.text[:200]}
            for c in result.only_in_b
        ],
    }

    logger.info(
        f"[CompareTask] 比对完成: "
        f"对齐 {result.stats.total_aligned}, "
        f"差异 {result.stats.with_differences}, "
        f"session={session_id[:8]}..."
    )

    return output
