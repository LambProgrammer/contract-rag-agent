"""
检索层召回评测（Retrieval Recall / MRR）
=========================================
在项目中的角色：
    补齐 RAG 评估体系中"检索层"的量化缺口。现有 `test_ragas.py` 评估的是
    端到端答案质量（Faithfulness / AnswerRelevancy / ContextRecall / ContextRelevance），
    属于 LLM 语义评判，无法定位问题出在"没召回"还是"没排好"。

    本脚本只测**检索层**，用确定性指标（hit@k / MRR），不受 LLM 评判波动影响，
    可复现、可回归。

为什么用条款号作为评测锚点：
    目标集合可以**客观唯一确定**——不需要人工标注"这题的相关文档是哪些"。
    只要 `clause_ref` 以指定条款号开头、且 `doc_id` 属于指定合同，即为相关。
    这避开了检索指标最大的落地障碍：主观标注。

两类查询（这是本评测的核心设计）：
    A 类｜纯条款号查询：只给条款号，不给合同限定
         目标 = 该条款号在**任意一份**合同中的 chunk
         预期：稀疏路（字面命中"第X条"）应优于稠密路

    B 类｜条款号 + 合同类型限定：既要条款号，又要指定合同
         目标 = 指定合同的该条款
         预期：单路都可能被"条款号相同但内容不同"的另一份合同干扰，
               需要两路互补才能稳定定位

    两类合起来，测的不是"哪个检索器更强"，而是**"为什么这个场景需要两路"**。

前置条件：
    - Qdrant 中已索引合同数据（至少含下述两份合同）
    - 本地已缓存 BGE / CrossEncoder 模型（同 test_ragas.py）

运行方式：
    uv run pytest tests/eval/test_retrieval_recall.py -v -s
    或：uv run python -m tests.eval.test_retrieval_recall

注意：
    本脚本**不修改任何业务代码**，只 import `src/retrieval/` 下已有的检索器。
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# ============================================================
# 评测配置
# ============================================================
# 参与评测的 top_k 档位——5 是最终送 LLM 的数量，20 是召回阶段的候选池
TOPK_LEVELS = (5, 10, 20)

# 两份条款号高度重合、但内容完全不同的合同（用于 B 类查询的消歧）
# doc_id 取自 Qdrant 中实际入库的记录（同一份合同被重复上传过多次）
DOC_IDS_TAIWAN = {
    "014b6e8b-a3fe-41da-b9b0-c633467b2b86",
    "b877928c-5fc7-463c-8177-01c29a74c54c",
}
DOC_IDS_YANXUE = {
    "0d004abf-3e08-49d6-8610-c2068b6bff31",
    "4106ee3a-e939-47d8-8028-855f93eabd5e",
}


@dataclass
class Query:
    """一条评测查询"""

    text: str
    """查询文本"""

    target_ref: str
    """目标条款号（用 startswith 匹配 clause_ref，容忍合同间空格差异）"""

    target_docs: set[str] | None = None
    """限定合同（doc_id 集合）；None 表示不限合同——即 A 类查询"""

    kind: str = "A"
    """A = 纯条款号；B = 条款号 + 合同限定"""


def build_queries() -> list[Query]:
    """
    构造评测查询集。

    8 个条款号在台湾合同与研学合同中**同时存在但内容不同**：
        第十六条  台湾=其他责任            / 研学=旅行社的违约责任
        第十四条  台湾=赴台游旅行社的违约责任 / 研学=必要的费用扣除
        第十五条  台湾=旅游者的违约责任     / 研学=旅行社协助旅游者返程及费用承担
        第十一条  台湾=旅游者解除合同       / 研学=旅行社解除合同
        第七条    台湾=赴台游旅行社的义务   / 研学=旅行社的义务
        第十三条  台湾=必要的费用扣除       / 研学=因不可抗力…解除合同
        第十二条  台湾=因不可抗力…解除合同  / 研学=旅游者解除合同
        第十七条  台湾=线路行程时间         / 研学=旅游者的违约责任
    """
    refs = [
        "第十六条",
        "第十四条",
        "第十五条",
        "第十一条",
        "第七条",
        "第十三条",
        "第十二条",
        "第十七条",
    ]

    queries: list[Query] = []

    # ---- A 类：只给条款号，不限合同 ----
    for ref in refs:
        queries.append(
            Query(text=f"合同里{ref}的内容是什么？", target_ref=ref, kind="A")
        )

    # ---- B 类：条款号 + 合同类型限定（交替覆盖两份合同）----
    for i, ref in enumerate(refs):
        if i % 2 == 0:
            queries.append(
                Query(
                    text=f"研学旅游合同的{ref}是怎么约定的？",
                    target_ref=ref,
                    target_docs=DOC_IDS_YANXUE,
                    kind="B",
                )
            )
        else:
            queries.append(
                Query(
                    text=f"赴台游旅行社旅游合同的{ref}是怎么约定的？",
                    target_ref=ref,
                    target_docs=DOC_IDS_TAIWAN,
                    kind="B",
                )
            )

    return queries


# ============================================================
# 结果数据结构
# ============================================================
@dataclass
class RetrieverScore:
    """单个检索器在单个 top_k 档位下的累计得分"""

    hits: int = 0
    total: int = 0
    reciprocal_ranks: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_ranks / self.total if self.total else 0.0


@dataclass
class EvalReport:
    """完整评测结果"""

    # {retriever_name: {kind: {k: RetrieverScore}}}
    scores: dict[str, dict[str, dict[int, RetrieverScore]]] = field(
        default_factory=dict
    )

    def add(
        self, retriever: str, kind: str, k: int, values: list[float | None]
    ) -> None:
        """
        记录一批查询的结果。

        参数：
            values: 每条查询的命中排名（1-based）；未命中为 None
        """
        s = (
            self.scores.setdefault(retriever, {})
            .setdefault(kind, {})
            .setdefault(k, RetrieverScore())
        )
        for rank in values:
            s.total += 1
            if rank is not None:
                s.hits += 1
                s.reciprocal_ranks += 1.0 / rank


# ============================================================
# 核心评测逻辑
# ============================================================
def _rank_of_first_hit(results: Any, query: Query) -> int | None:
    """
    在一批检索结果中找"第一个命中的目标"的排名（1-based）。

    命中判定：
        results[i].clause_ref 以 query.target_ref 开头
        且（若指定了合同）results[i].doc_id 属于 query.target_docs
    """
    for i, doc in enumerate(results, 1):
        ref = getattr(doc, "clause_ref", "") or ""
        if not ref.startswith(query.target_ref):
            continue
        if query.target_docs is not None:
            if getattr(doc, "doc_id", "") not in query.target_docs:
                continue
        return i
    return None


def run_evaluation(with_rerank: bool = True) -> EvalReport:
    """
    执行检索层评测。

    参数：
        with_rerank: 是否额外评测"混合检索 + CrossEncoder 精排"（即生产管线形态）

    返回：
        EvalReport，含各检索器在 A/B 两类查询、各 top_k 档位下的 hit@k 与 MRR
    """
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.sparse_retriever import SparseRetriever

    queries = build_queries()
    max_k = max(TOPK_LEVELS)

    # ---- 初始化检索器（模型为延迟加载，这里只创建对象）----
    retrievers: dict[str, Any] = {
        "dense": DenseRetriever(),
        "sparse": SparseRetriever(),
        "hybrid": HybridRetriever(),
    }
    reranker = None
    if with_rerank:
        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

    report = EvalReport()

    for qi, q in enumerate(queries, 1):
        logger.info(f"[RetrievalEval] ({qi}/{len(queries)}) [{q.kind}] {q.text}")

        for name, retriever in retrievers.items():
            try:
                docs = retriever.retrieve(q.text, top_k=max_k)
            except Exception as e:  # noqa: BLE001
                logger.error(f"  {name} 检索失败: {e}")
                docs = []

            # 三个档位都从同一份结果里截取，避免重复检索
            per_k: dict[int, float | None] = {}
            for k in TOPK_LEVELS:
                per_k[k] = _rank_of_first_hit(docs[:k], q)
            for k in TOPK_LEVELS:
                report.add(name, q.kind, k, [per_k[k]])

            logger.info(
                f"    {name:<7} hit@5={'Y' if per_k[5] else 'N'} "
                f"rank={per_k[5] if per_k[5] else '-'}"
            )

        # ---- 生产管线形态：混合检索 → CrossEncoder 精排 ----
        if reranker is not None:
            try:
                base = retrievers["hybrid"].retrieve(q.text, top_k=max_k)
                reranked = reranker.rerank(q.text, list(base))
            except Exception as e:  # noqa: BLE001
                logger.error(f"  rerank 失败: {e}")
                reranked = []

            per_k = {}
            for k in TOPK_LEVELS:
                per_k[k] = _rank_of_first_hit(reranked[:k], q)
            for k in TOPK_LEVELS:
                report.add("hybrid+rerank", q.kind, k, [per_k[k]])

            logger.info(
                f"    {'hyb+rr':<7} hit@5={'Y' if per_k[5] else 'N'} "
                f"rank={per_k[5] if per_k[5] else '-'}"
            )

    return report


# ============================================================
# 报告输出
# ============================================================
KIND_LABEL = {"A": "A类 纯条款号查询", "B": "B类 条款号+合同限定", "ALL": "合计"}


def _aggregate(scores: dict[str, dict[int, RetrieverScore]], k: int) -> RetrieverScore:
    """把 A/B 两类合并成一个总分"""
    total = RetrieverScore()
    for kind_scores in scores.values():
        if k in kind_scores:
            s = kind_scores[k]
            total.hits += s.hits
            total.total += s.total
            total.reciprocal_ranks += s.reciprocal_ranks
    return total


def print_report(report: EvalReport) -> None:
    """终端输出评测报告表"""
    retrievers = list(report.scores.keys())
    width = 18

    print("\n" + "=" * 78)
    print("  检索层召回评测报告  (hit@k / MRR，目标由 clause_ref + doc_id 客观确定)")
    print("=" * 78)

    for kind in ("A", "B", "ALL"):
        print(f"\n【{KIND_LABEL[kind]}】")
        header = f"  {'检索器':<{width}}" + "".join(
            f"{'hit@' + str(k):>10}{'MRR':>8}" for k in TOPK_LEVELS
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in retrievers:
            row = f"  {name:<{width}}"
            for k in TOPK_LEVELS:
                if kind == "ALL":
                    s = _aggregate(report.scores[name], k)
                else:
                    s = report.scores[name].get(kind, {}).get(k, RetrieverScore())
                row += f"{s.hit_rate:>9.1%} {s.mrr:>8.3f}"
            print(row)

    print("\n" + "=" * 78)
    print("  说明：hit@k = 目标条款是否出现在前 k 条；MRR = 首次命中排名的倒数均值")
    print("=" * 78 + "\n")


# ============================================================
# pytest 入口
# ============================================================
@pytest.mark.slow
@pytest.mark.eval
def test_retrieval_recall():
    """
    检索层召回评测。

    注意：本测试依赖 Qdrant 中已索引的合同数据。
    如 Qdrant 不可达或数据缺失，会跳过（不阻塞 CI）。
    """
    try:
        report = run_evaluation()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"检索评测前置条件不满足（Qdrant 或模型不可用）: {e}")
        return

    print_report(report)

    # 不作硬断言（基线数据会波动，避免因数据缺失失败），只校验结构合理
    assert report.scores, "评测未产生任何结果"


# ============================================================
# 直接运行入口
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s"
    )
    print_report(run_evaluation())
    sys.exit(0)
