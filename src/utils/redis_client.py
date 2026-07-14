"""
Redis 工具模块（Chat History）
===============================
在 RAG 系统中的角色：
    存储多轮对话历史，支持 rewrite_node 消解指代
    （"这个条款合法吗"→"第十六条 违约责任条款的合法性"）。

为什么用 Redis 而非 PostgreSQL：
    - 对话历史是临时的（1 小时 TTL 后自动清除），不需要持久化
    - 读写频率高（每轮 QA 一次读、一次写），Redis 比 PG 快 100x
    - Redis 已经在基础设施中（Celery broker），零额外部署成本

存储格式：
    Key:   chat:{session_id}
    Value: JSON 数组 [{role, content}, ...]
    TTL:   3600 秒（与 Celery 结果过期时间一致）

用法：
    from src.utils.redis_client import get_history, append_history
    history = get_history(session_id)  # → [{"role":"user","content":"..."}, ...]
    append_history(session_id, [{"role":"user", "content": query},
                                 {"role":"assistant", "content": answer}])
"""

import json
import logging
from typing import List

import redis

from src.core.config import settings

logger = logging.getLogger(__name__)

# 保留最近 N 轮对话（1 轮 = user + assistant），避免 prompt 过长
MAX_ROUNDS = 10
# Redis key 的 TTL（秒）
HISTORY_TTL = 3600
# Redis key 前缀
KEY_PREFIX = "chat:"

# 共享连接（由 FastAPI lifespan 注入，Worker 进程保持 None）
_shared_redis: "redis.Redis | None" = None


def set_shared_redis(client: "redis.Redis") -> None:
    """
    注入全局共享 Redis 连接。

    由 FastAPI lifespan startup 调用。Worker 进程不调用此函数，
    因此 _shared_redis 在 Worker 中保持 None，_get_redis() 会自建连接。
    """
    global _shared_redis
    _shared_redis = client


def _get_redis() -> redis.Redis:
    """
    获取 Redis 连接。

    优先使用 lifespan 注入的共享连接（FastAPI 进程），
    无共享连接时自建（Worker 进程 / 无 lifespan 的上下文）。
    """
    if _shared_redis is not None:
        return _shared_redis
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


def get_history(session_id: str) -> List[dict]:
    """
    从 Redis 读取对话历史。

    返回：
        [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "yyy"}, ...]
        如果没有历史记录，返回空列表。
    """
    r = _get_redis()
    try:
        data = r.get(f"{KEY_PREFIX}{session_id}")
        if data:
            return json.loads(data)
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.warning(f"[Redis] 读取对话历史失败: {e}")
    return []


def append_history(session_id: str, turns: List[dict]) -> None:
    """
    将新的对话轮次追加到 Redis，并截断超出的轮次。

    参数：
        session_id: 会话 ID
        turns: 新轮次的 [{role, content}, ...]
    """
    r = _get_redis()
    try:
        key = f"{KEY_PREFIX}{session_id}"
        existing = get_history(session_id)
        existing.extend(turns)

        # 保留最近 MAX_ROUNDS 轮（1 轮 = 2 条消息）
        max_messages = MAX_ROUNDS * 2
        if len(existing) > max_messages:
            existing = existing[-max_messages:]

        r.set(key, json.dumps(existing, ensure_ascii=False), ex=HISTORY_TTL)
        logger.info(
            f"[Redis] 已保存对话历史: session={session_id[:8]}..., "
            f"{len(existing)} 条消息 ({len(existing) // 2} 轮)"
        )
    except redis.RedisError as e:
        logger.warning(f"[Redis] 保存对话历史失败: {e}")


def history_to_text(history: List[dict]) -> str:
    """
    将对话历史转为 rewrite_node 可用的纯文本格式。

    格式：
        用户：违约责任怎么计算？
        系统：根据第十六条...
        用户：这个条款合法吗？

    用于在 rewrite prompt 中提供上下文以消解指代。
    """
    if not history:
        return "（无对话历史）"

    lines = []
    for msg in history:
        role_label = "用户" if msg["role"] == "user" else "系统"
        content = msg.get("content", "")[:200]  # 截断过长的答案
        lines.append(f"{role_label}：{content}")
    return "\n".join(lines)


def get_compare_context(session_id: str) -> str | None:
    """
    读取比对/风险检测快照（如存在），转为 rewrite_node 可用的上下文。

    返回：
        格式化后的上下文文本，如无则返回 None
    """
    r = _get_redis()
    try:
        data = r.get(f"compare:{session_id}")
        if data:
            info = json.loads(data)

            # risk 快照
            if info.get("type") == "risk":
                risks = info.get("risks", [])
                if not risks:
                    return None
                risk_lines = [
                    f"  - [{r['rule_id']}] {r['rule_name']} ({r['severity']})\n"
                    f"    条款：{r['clause_ref']}\n"
                    f"    分析：{r.get('analysis', '')}\n"
                    f"    建议：{r.get('suggestion', '')}"
                    for r in risks
                ]
                return (
                    f"[风险检测结果] 此前已对合同进行风险检测，"
                    f"共发现 {len(risks)} 条风险：\n\n"
                    + "\n\n".join(risk_lines)
                    + "\n\n用户的后续提问可能指代这些风险中的某一条。"
                )

            # compare 快照
            stats = info.get("stats", {})
            diffs = info.get("differences", [])
            diff_lines = []
            for d in diffs:
                diff_lines.append(
                    f"  - {d['clause_ref_a']} vs {d['clause_ref_b']}：\n"
                    f"    {d['analysis']}"
                )
            diff_text = "\n".join(diff_lines) if diff_lines else ""
            return (
                f"[比对上下文] 此前已对两份合同进行了条款比对：\n"
                f"  - 对齐条款数：{stats.get('total_aligned', '?')}\n"
                f"  - 完全一致：{stats.get('identical', '?')}\n"
                f"  - 存在差异：{stats.get('with_differences', '?')}\n"
                f"  - 仅甲方有：{stats.get('only_in_a', '?')}\n"
                f"  - 仅乙方有：{stats.get('only_in_b', '?')}\n"
                + (f"\n差异详情：\n{diff_text}\n" if diff_text else "")
                + "\n用户的后续提问可能指代这些差异中的某一条。"
            )
    except Exception:
        pass
    return None
