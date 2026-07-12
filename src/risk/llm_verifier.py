"""
LLM 风险二次确认器
==================
在 RAG 系统中的角色：
    接收 rule_engine 的关键词命中结果，逐条让 LLM 判断
    "这真的是风险吗，还是关键词恰好出现的误报？"

为什么需要 LLM 二次确认（不能纯靠关键词）：
    关键词匹配只看"词在不在文本里"，不看"这个词在这个语境下是什么意思"。
    例如：合同里写"双方协商确定违约金比例"——包含"违约金"关键词，
    但这不是风险条款，只是提到了这个词。如果纯靠关键词报风险，
    用户会对系统失去信任。LLM 能理解语境，过滤掉这类误报。

LLM 的判断任务：
    1. 这条条款确实构成了风险吗？（是/否/部分）
    2. 如果是，风险的严重程度是否和规则标注一致？
    3. 给出具体的风险分析（简要说明为什么需要关注）

Prompt 设计：
    系统指令：告诉 LLM 它的角色是合同风险审核员
    上下文：规则描述 + 条款原文 + 触发关键词
    输出：判定结果 + 理由

用法：
    from src.risk.llm_verifier import verify_with_llm
    verified = verify_with_llm(hits)
    # → List[VerifiedRisk]: 经 LLM 确认的最终风险列表
"""

import logging
from dataclasses import dataclass
from typing import List

from pydantic import SecretStr

from src.core.config import settings
from src.risk.rule_engine import RiskHit

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================
@dataclass
class VerifiedRisk:
    """经 LLM 确认的风险条目"""

    rule_id: str
    """对应 risk_rules.yaml 中的规则 ID"""

    rule_name: str
    """规则名称"""

    severity: str
    """原始严重程度（LLM 可能调高/调低）"""

    clause_ref: str
    """命中的条款引用"""

    chunk_text: str
    """条款原文片段（截取前 300 字符）"""

    matched_keyword: str
    """触发关键词"""

    verdict: str
    """LLM 判定结果：confirmed（确认风险）/ false_alarm（误报）/ adjusted（确认但调整严重程度）"""

    analysis: str
    """LLM 给出的风险分析（为什么需要关注/为什么是误报）"""

    suggestion: str
    """建议操作（优先 LLM 给出的，回退到规则自带的 suggestion）"""


# ============================================================
# LLM 确认逻辑
# ============================================================
def verify_with_llm(hits: List[RiskHit]) -> List[VerifiedRisk]:
    """
    对关键词命中的条目逐条做 LLM 二次确认。

    参数：
        hits: rule_engine 输出的命中列表

    返回：
        经 LLM 确认的风险列表（误报已被过滤）

    注意：
        当前为串行调用（逐条发送给 LLM）。如果命中数超过 20 条，
        考虑批量化（一次 prompt 送多条）以节省 API 延迟。M5 优化。
    """
    from langchain_deepseek import ChatDeepSeek

    if not hits:
        return []

    logger.info(f"[Verifier] LLM 确认 {len(hits)} 条命中...")

    llm = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=SecretStr(settings.deepseek_api_key),
        temperature=0.2,
    )

    verified = []
    for i, hit in enumerate(hits, 1):
        result = _verify_single(hit, llm, i, len(hits))
        if result.verdict != "false_alarm":
            verified.append(result)

    confirmed = sum(1 for v in verified if v.verdict != "false_alarm")
    false_alarms = len(hits) - len(verified)
    logger.info(
        f"[Verifier] LLM 确认完成: {confirmed} 条真实风险, {false_alarms} 条误报已过滤"
    )

    return verified


def _verify_single(hit: RiskHit, llm, idx: int, total: int) -> VerifiedRisk:
    """对单条命中做 LLM 确认"""
    logger.info(f"[Verifier] ({idx}/{total}) 检查: [{hit.rule.id}] {hit.rule.name}")

    system_prompt = (
        "你是一个专业的合同风险审核员。请阅读以下合同条款，"
        "判断它是否构成风险检测规则所描述的风险。\n\n"
        "请按以下格式回答：\n"
        "判定: [confirmed / false_alarm / adjusted]\n"
        "分析: [简要说明为什么这是/不是风险]\n"
        "建议: [如果确认是风险，给出修改建议；如果是误报，写'无']\n\n"
        "判定标准（请严格遵循）：\n"
        "- confirmed: 条款内容明显偏离行业惯例或法律合理范围，存在实质风险，需要关注\n"
        "- false_alarm: 触发以下任一条件即判定为误报——\n"
        "  (a) 关键词出现了但条款本身没有风险（例如合同中写'违约金由双方协商确定'——包含'违约金'但并未约定过高比例）\n"
        "  (b) 条款约定的内容属于行业通行惯例或合理范围，即使关键词命中，也不应视为风险\n"
        "      （例如违约金比例为20%且双方对等、不可抗力范围与民法典规定一致、\n"
        "        仲裁机构名称完整且约定清晰、付款条件以客观验收为基准等）\n"
        "  (c) 条款内容双向对等且比例处于合理区间内\n"
        "- adjusted: 存在风险但严重程度与规则标注不同",
    )

    user_prompt = (
        f"=== 检测规则 ===\n"
        f"规则名称: {hit.rule.name}\n"
        f"严重程度: {hit.rule.severity}\n"
        f"风险描述: {hit.rule.description}\n"
        f"触发关键词: {hit.matched_keyword}\n\n"
        f"=== 合同条款原文 ===\n"
        f"条款引用: {hit.chunk.clause_ref}\n"
        f"{hit.chunk.text[:500]}\n"
    )

    response = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
    content = str(response.content)

    # 解析 LLM 输出
    verdict = "confirmed"
    analysis = content
    suggestion = hit.rule.suggestion

    for line in content.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("判定:") or line_stripped.startswith("判定："):
            raw = line_stripped.split(":", 1)[-1].split("：", 1)[-1].strip().lower()
            if "false" in raw:
                verdict = "false_alarm"
            elif "adjusted" in raw:
                verdict = "adjusted"
        elif line_stripped.startswith("分析:") or line_stripped.startswith("分析："):
            analysis = line_stripped.split(":", 1)[-1].split("：", 1)[-1].strip()
        elif line_stripped.startswith("建议:") or line_stripped.startswith("建议："):
            sug = line_stripped.split(":", 1)[-1].split("：", 1)[-1].strip()
            if sug and sug != "无":
                suggestion = sug

    return VerifiedRisk(
        rule_id=hit.rule.id,
        rule_name=hit.rule.name,
        severity=hit.rule.severity,
        clause_ref=hit.chunk.clause_ref,
        chunk_text=hit.chunk.text[:300],
        matched_keyword=hit.matched_keyword,
        verdict=verdict,
        analysis=analysis,
        suggestion=suggestion,
    )
