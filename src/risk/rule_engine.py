"""
风险规则引擎
============
在 RAG 系统中的角色：
    加载 risk_rules.yaml，遍历合同 Chunks，用关键词匹配筛选出
    可能包含风险的条款。命中结果送入 llm_verifier.py 做 LLM 二次确认。

为什么用关键词匹配而非 LLM 全量扫描：
    - 性能：30 个 Chunks × 13 条规则 = 390 次关键词检查（毫秒级）
      vs 30 次 LLM 调用（分钟级）
    - 成本：关键词匹配 0 token vs LLM 全量扫描 ~150K tokens
    - 策略：关键词做"粗筛"，LLM 做"精判"——只有命中的才送 LLM

关键词匹配的局限性（有意为之）：
    - 可能漏报（规则未覆盖的风险）——这是关键词引擎的固有局限，
      但不影响"已命中规则"的检测质量
    - 可能误报（关键词出现了但实际不是风险）——由 llm_verifier 过滤
    - 规则库可通过 YAML 持续扩充，不涉及代码修改

用法：
    from src.risk.rule_engine import RiskRuleEngine
    engine = RiskRuleEngine()
    hits = engine.scan(doc_id="abc", chunks=[...])
    # → List[RiskHit]: 按严重程度排序的命中列表
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from src.chunkers.legal_chunker import Chunk

logger = logging.getLogger(__name__)

# 规则文件路径（相对于项目根目录）
RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "risk_rules.yaml"

# 严重程度排序权重（用于结果排序）
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ============================================================
# 数据模型
# ============================================================
@dataclass
class RiskRule:
    """一条风险检测规则"""

    id: str
    name: str
    keywords: List[str]
    severity: str
    description: str
    suggestion: str


@dataclass
class RiskHit:
    """
    一条规则在一个 Chunk 上的命中记录。

    这是 rule_engine 的输出单位——每个 Hit 代表
    "某条规则的关键词在某个 Chunk 的文本中被找到"。
    """

    rule: RiskRule
    """命中的规则"""

    chunk: Chunk
    """命中的 Chunk（包含条款原文、clause_ref、doc_id）"""

    matched_keyword: str
    """触发命中的具体关键词（用于 LLM 二次确认时的上下文）"""


# ============================================================
# 规则引擎
# ============================================================
class RiskRuleEngine:
    """
    风险规则引擎。

    从 YAML 加载规则，对 Chunk 列表做关键词扫描。
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        """
        初始化引擎，加载规则。

        参数：
            rules_path: 规则文件路径，默认 config/risk_rules.yaml
        """
        self._rules_path = rules_path or RULES_PATH
        self._rules: List[RiskRule] = []
        self._load_rules()

    # ----------------------------------------------------------
    # 加载规则
    # ----------------------------------------------------------
    def _load_rules(self) -> None:
        """从 YAML 文件加载风险规则"""
        if not self._rules_path.exists():
            logger.error(f"[RuleEngine] 规则文件不存在: {self._rules_path}")
            return

        with open(self._rules_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for item in data.get("rules", []):
            rule = RiskRule(
                id=item["id"],
                name=item["name"],
                keywords=item.get("keywords", []),
                severity=item.get("severity", "medium"),
                description=item.get("description", ""),
                suggestion=item.get("suggestion", ""),
            )
            self._rules.append(rule)

        logger.info(f"[RuleEngine] 加载了 {len(self._rules)} 条风险规则")

    # ----------------------------------------------------------
    # 扫描
    # ----------------------------------------------------------
    def scan(self, doc_id: str, chunks: List[Chunk]) -> List[RiskHit]:
        """
        对合同的全部 Chunks 执行规则扫描。

        参数：
            doc_id: 文档 ID（用于日志追踪）
            chunks: 待扫描的 Chunk 列表

        返回：
            命中列表，按严重程度（critical → low）排序；
            同一严重等级按规则 ID 排序。
        """
        if not self._rules:
            logger.warning("[RuleEngine] 无可用规则，跳过扫描")
            return []

        logger.info(
            f"[RuleEngine] 开始扫描: doc_id={doc_id[:8]}..., "
            f"{len(chunks)} chunks × {len(self._rules)} rules"
        )

        hits = []
        for chunk in chunks:
            for rule in self._rules:
                matched = self._match_chunk(chunk, rule)
                if matched:
                    hits.append(
                        RiskHit(
                            rule=rule,
                            chunk=chunk,
                            matched_keyword=matched,
                        )
                    )

        # 按严重程度排序：critical → high → medium → low
        hits.sort(key=lambda h: (SEVERITY_ORDER.get(h.rule.severity, 99), h.rule.id))

        logger.info(
            f"[RuleEngine] 扫描完成: {len(hits)} 次命中 "
            f"（命中率 {len(hits) / max(len(chunks) * len(self._rules), 1) * 100:.1f}%）"
        )

        # 打印命中摘要
        if hits:
            severity_counts = {}
            for h in hits:
                sev = h.rule.severity
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            logger.info(f"[RuleEngine] 命中分布: {severity_counts}")

        return hits

    # ----------------------------------------------------------
    # 单 Chunk 匹配
    # ----------------------------------------------------------
    def _match_chunk(self, chunk: Chunk, rule: RiskRule) -> str | None:
        """
        检查单个 Chunk 是否命中某条规则的关键词。

        返回：
            命中的关键词（取第一个命中的），或 None（未命中）
        """
        for kw in rule.keywords:
            if kw in chunk.text:
                return kw
        return None
