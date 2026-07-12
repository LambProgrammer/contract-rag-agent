"""
风险检测异步任务（process_risk_detection）
==========================================
在 RAG 系统中的角色：
    将 M4 的同步风险检测端点改造为 Celery 异步任务。
    解决长耗时检测（大量 LLM 调用）导致 HTTP 网关超时的问题。

与 process_contract 相同的异步模式：
    API 端 → task.delay() → 秒返 task_id → Worker 后台执行 → 前端轮询进度

用法：
    from src.tasks.risk_task import process_risk_detection
    task = process_risk_detection.delay(doc_id="xxx")
"""

import json
import logging

from celery import shared_task

from src.core.config import settings

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=1, name="process_risk_detection")
def process_risk_detection(self, doc_id: str) -> dict:
    """
    异步执行风险条款检测。

    参数：
        doc_id: 已入库合同的文档 ID

    返回：
        {"doc_id": ..., "total_risks": ..., "keyword_hits": ..., "session_id": ...}
    """
    self.update_state(state="PROCESSING", meta={"progress": 10, "stage": "风险检测"})

    logger.info(f"[RiskTask] 开始检测: doc_id={doc_id[:8]}...")

    # ---- 从 Qdrant 拉取 Chunks ----
    from src.indexing.vector_client import QdrantVectorClient

    qdrant = QdrantVectorClient()
    from src.risk.router import _get_chunks_by_doc_id

    chunks = _get_chunks_by_doc_id(qdrant, doc_id)
    if not chunks:
        raise ValueError(f"未找到文档 {doc_id} 的索引数据")

    self.update_state(
        state="PROCESSING",
        meta={"progress": 30, "stage": "关键词扫描", "doc_id": doc_id},
    )

    # ---- 关键词规则引擎扫描 ----
    from src.risk.rule_engine import RiskRuleEngine

    engine = RiskRuleEngine()
    hits = engine.scan(doc_id, chunks)

    self.update_state(
        state="PROCESSING",
        meta={
            "progress": 60,
            "stage": f"LLM 确认 {len(hits)} 条命中",
            "doc_id": doc_id,
        },
    )

    # ---- LLM 二次确认 ----
    from src.risk.llm_verifier import verify_with_llm

    verified_risks = verify_with_llm(hits)

    # ---- 生成 session_id 并存入 Redis ----
    import uuid

    session_id = str(uuid.uuid4())
    try:
        import redis as rds

        client = rds.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=False,
        )
        data = json.dumps(
            {
                "type": "risk",
                "doc_id": doc_id,
                "total_risks": len(verified_risks),
                "risks": [
                    {
                        "rule_id": r.rule_id,
                        "rule_name": r.rule_name,
                        "severity": r.severity,
                        "clause_ref": r.clause_ref,
                        "analysis": r.analysis,
                    }
                    for r in verified_risks
                ],
            },
            ensure_ascii=False,
        )
        client.set(f"compare:{session_id}", data, ex=3600)
    except Exception as e:
        logger.warning(f"[RiskTask] Redis 存储失败: {e}")

    result = {
        "doc_id": doc_id,
        "total_chunks": len(chunks),
        "rules_loaded": len(engine._rules),
        "keyword_hits": len(hits),
        "verified_risks": len(verified_risks),
        "false_alarms_filtered": len(hits) - len(verified_risks),
        "session_id": session_id,
        "risks": [
            {
                "rule_id": r.rule_id,
                "rule_name": r.rule_name,
                "severity": r.severity,
                "clause_ref": r.clause_ref,
                "chunk_text": r.chunk_text,
                "matched_keyword": r.matched_keyword,
                "verdict": r.verdict,
                "analysis": r.analysis,
                "suggestion": r.suggestion,
            }
            for r in verified_risks
        ],
    }

    logger.info(
        f"[RiskTask] 检测完成: {len(verified_risks)} 条风险, "
        f"session={session_id[:8]}..."
    )

    return result
