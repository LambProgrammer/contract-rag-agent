"""
RAG 系统 Ragas 评估脚本
========================
在 RAG 系统中的角色：
    使用 Ragas 四项核心指标（Faithfulness / AnswerRelevancy /
    ContextRecall / ContextRelevance）对 RAG 管线进行量化评估，
    并将每项得分上报 LangFuse 以便持续追踪质量变化。

评估流程：
    1. 加载 JSON 测试集（question + ground_truth）
    2. 逐条调用 RAG 图获取 answer + retrieved_contexts
    3. 组装为 Ragas EvaluationDataset
    4. 使用 DeepSeek 作为评估 LLM，计算 4 项指标
    5. 终端输出分数表，同步上报 LangFuse

前置条件：
    - Qdrant 中已有已索引的合同数据（先上传一份合同并等待解析完成）
    - .env 中 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 已配置

运行方式：
    uv run pytest tests/eval/test_ragas.py -v -s
    或直接：
    uv run python -m tests.eval.test_ragas

评估 LLM：
    使用 DeepSeek（langchain-deepseek）作为 Ragas 的评判 LLM。
    Ragas 用评判 LLM 来判断答案是否忠实于检索上下文、是否切题等。
    不同评判 LLM 的绝对分值可能不同，因此相对变化比绝对分值更重要。
"""

import json
import logging
import os
import sys
import time
from typing import Any

import pytest

# ---- Ragas 0.4.3 兼容补丁（必须在 import ragas 之前调用）----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.ragas_compat import apply_ragas_compat  # noqa: E402

apply_ragas_compat()

# ---- 第三方库 ----
from langfuse import Langfuse  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import BaseRagasEmbedding  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402  -- 使用旧版 API（新版 collections 不兼容 aevaluate）
    answer_relevancy,
    context_recall,
    faithfulness,
)
from ragas.metrics._nv_metrics import ContextRelevance  # noqa: E402

from src.core.config import settings  # noqa: E402

logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================
TEST_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "test_questions.json")
EVAL_MODEL = "deepseek-chat"  # DeepSeek 模型用于 Ragas 评判
EVAL_TEMPERATURE = 0.1  # 评判用低温，确保一致性


# ============================================================
# BGE Embedding 封装（Ragas 内置 HuggingFaceEmbeddings 与
# 新版 sentence-transformers 存在 kwargs 兼容问题，故自定义薄封装）
# ============================================================
class _BGERagasEmbedding(BaseRagasEmbedding):
    """将 BGEEmbedder 包装为 Ragas 兼容的 embedding provider"""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._dim = self._model.get_embedding_dimension()  # type: ignore[reportUnknownMemberType]

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        result = self._model.encode(  # type: ignore[reportUnknownMemberType]
            [text], normalize_embeddings=True, **kwargs
        )
        return result[0].tolist()

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text, **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """旧版 ragas 指标调用此方法嵌入单个查询"""
        return self.embed_text(text, **kwargs)

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """旧版 ragas 指标调用此方法批量嵌入文档"""
        embeddings = self._model.encode(  # type: ignore[reportUnknownMemberType]
            texts, normalize_embeddings=True, **kwargs
        )
        return embeddings.tolist()


# ============================================================
# 辅助函数
# ============================================================
def _load_test_questions() -> list[dict[str, Any]]:
    """加载 JSON 测试集"""
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    logger.info(f"[Eval] 加载 {len(questions)} 条测试问题")
    return questions


def _run_rag_graph(
    question: str, history_text: str = "", doc_id: str = ""
) -> dict[str, Any]:
    """
    运行一次 RAG 图，返回 answer 和 retrieved_contexts。

    参数：
        question: 用户提问
        history_text: 对话历史文本（多轮消解指代用）
        doc_id: 限定合同范围（空 = 全部合同）

    返回：
        {"answer": str, "contexts": [str, ...]}
    """

    from src.graph.graph_builder import build_rag_graph  # type: ignore[import-not-found]
    from src.graph.state import RAGState  # type: ignore[import-not-found]

    graph = build_rag_graph()

    initial_state: RAGState = {
        "query": question,
        "session_id": f"eval-{os.urandom(4).hex()}",
        "history": history_text,
        "doc_id": doc_id,
        "rewritten_query": "",
        "retrieved_docs": [],
        "answer": "",
    }

    result = graph.invoke(initial_state)

    # 提取检索上下文
    retrieved_docs = result.get("retrieved_docs", [])
    contexts = [doc.text for doc in retrieved_docs if hasattr(doc, "text")]

    return {
        "answer": result.get("answer", ""),
        "contexts": contexts,
        "rewritten_query": result.get("rewritten_query", ""),
    }


def _build_evaluator_llm() -> Any:  # noqa: ANN401
    """构建 Ragas 评判 LLM（DeepSeek，LangChain 兼容接口）"""
    from langchain_deepseek import ChatDeepSeek  # type: ignore[import-not-found]

    return LangchainLLMWrapper(
        ChatDeepSeek(
            model=EVAL_MODEL,
            api_key=settings.deepseek_api_key,  # type: ignore[reportArgumentType]
            temperature=EVAL_TEMPERATURE,
        )
    )


def _build_evaluator_embeddings() -> BaseRagasEmbedding:
    """构建 Ragas 评判用的 Embedding（项目同款 BGE 模型）"""
    return _BGERagasEmbedding(settings.embedding_model)


# ============================================================
# 评估主函数
# ============================================================
def run_evaluation() -> dict[str, Any]:
    """
    执行完整评估流程。

    返回：
        {
            "scores": {
                "faithfulness": float,
                "answer_relevancy": float,
                "context_recall": float,
                "context_relevance": float,
            },
            "samples": int,
            "duration_seconds": float,
        }
    """
    logger.info("[Eval] ========== Ragas 评估开始 ==========")

    # ---- 1. 加载测试集 ----
    questions = _load_test_questions()

    # ---- 2. 逐条运行 RAG 管线 ----
    samples = []
    for i, q in enumerate(questions):
        question = q["question"]
        ground_truth = q.get("ground_truth", "")
        category = q.get("category", "未分类")
        logger.info(f"[Eval] [{i + 1}/{len(questions)}] {category}: {question[:50]}...")

        try:
            # 处理多轮对话（需要上下文消解指代）
            history_text = ""
            if q.get("requires_history") and q.get("history_query"):
                prev_result = _run_rag_graph(q["history_query"], history_text="")
                history_text = (
                    f"用户：{q['history_query']}\n系统：{prev_result['answer']}"
                )
                logger.debug(f"  多轮历史: {q['history_query'][:40]}...")

            result = _run_rag_graph(question, history_text=history_text)

            if not result["answer"]:
                logger.warning("  跳过: 未生成答案")
                continue

            samples.append(
                SingleTurnSample(
                    user_input=question,
                    response=result["answer"],
                    retrieved_contexts=result["contexts"],
                    reference=ground_truth,
                )
            )
            logger.debug(
                f"  答案 {len(result['answer'])} 字符, "
                f"{len(result['contexts'])} 条上下文"
            )

        except Exception as e:
            logger.error(f"  失败: {e}", exc_info=True)
            continue

    if len(samples) < 4:
        raise RuntimeError(
            f"有效样本数 {len(samples)} 不足（至少需要 4 条），请检查测试数据"
        )

    logger.info(f"[Eval] 有效样本: {len(samples)}/{len(questions)}")

    # ---- 3. 构建 Ragas Dataset 并评估 ----
    dataset = EvaluationDataset(samples=samples)  # type: ignore[reportUnknownVariableType]
    evaluator_llm = _build_evaluator_llm()
    evaluator_embeddings = _build_evaluator_embeddings()
    # 使用旧版 ragas 指标（预先实例化的单例，继承 Metric 基类）
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        ContextRelevance(),
    ]

    logger.info("[Eval] 正在计算指标（调用 DeepSeek 评判，预计 2-5 分钟）...")
    start = time.perf_counter()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
    )

    elapsed = time.perf_counter() - start
    logger.info(f"[Eval] 评估完成，耗时 {elapsed:.1f}s")

    # ---- 4. 提取分数 ----
    # ragas evaluate() 返回值可能为 float（成功）或 list[float]（多样本聚合）
    def _safe_float(value: Any) -> float:
        """从 ragas 返回值中提取浮点数，兼容 list 和 NaN"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list) and len(value) > 0:
            valid = [float(v) for v in value if not _is_nan(v)]
            if valid:
                return sum(valid) / len(valid)
        return float("nan")

    def _is_nan(value: Any) -> bool:
        """检查值是否为 NaN"""
        try:
            return value != value  # type: ignore[reportUnnecessaryComparison]
        except Exception:
            return True

    _r: Any = result  # type: ignore[reportUnknownVariableType]
    scores: dict[str, float] = {
        "faithfulness": round(_safe_float(_r["faithfulness"]), 4),  # type: ignore[reportUnknownArgumentType]
        "answer_relevancy": round(_safe_float(_r["answer_relevancy"]), 4),  # type: ignore[reportUnknownArgumentType]
        "context_recall": round(_safe_float(_r["context_recall"]), 4),  # type: ignore[reportUnknownArgumentType]
        # 旧版 _ContextRelevance 在 result 中的 key 是 nv_context_relevance
        "context_relevance": round(_safe_float(_r["nv_context_relevance"]), 4),  # type: ignore[reportUnknownArgumentType]
    }

    # ---- 5. 上报 LangFuse ----
    _report_to_langfuse(scores, len(samples), elapsed)

    # ---- 6. 输出结果表 ----
    _print_scores(scores, len(samples), elapsed)

    return {
        "scores": scores,
        "samples": len(samples),
        "duration_seconds": elapsed,
    }


def _report_to_langfuse(
    scores: dict[str, float], sample_count: int, elapsed: float
) -> None:
    """将评估分数上报到 LangFuse"""
    try:
        langfuse = Langfuse()
        with langfuse.start_as_current_observation(
            name="ragas-evaluation",
            as_type="evaluator",
            input={
                "sample_count": sample_count,
                "metrics": list(scores.keys()),
            },
            output=scores,
            metadata={
                "evaluator_model": EVAL_MODEL,
                "duration_seconds": round(elapsed, 1),
            },
        ):
            for metric_name, value in scores.items():
                langfuse.score_current_span(name=metric_name, value=value)

        langfuse.flush()
        logger.info("[Eval] 分数已上报 LangFuse")
    except Exception as e:
        logger.warning(f"[Eval] LangFuse 上报失败（不影响评估结果）: {e}")


def _print_scores(scores: dict[str, float], sample_count: int, elapsed: float) -> None:
    """终端输出评估分数表"""
    print("\n" + "=" * 60)
    print("  Ragas RAG 质量评估报告")
    print("=" * 60)
    print(f"  评估样本数:     {sample_count}")
    print(f"  评判模型:       DeepSeek ({EVAL_MODEL})")
    print(f"  耗时:           {elapsed:.1f}s")
    print("-" * 60)
    print(f"  Faithfulness:        {scores['faithfulness']:.4f}")
    print(f"  AnswerRelevancy:     {scores['answer_relevancy']:.4f}")
    print(f"  ContextRecall:       {scores['context_recall']:.4f}")
    print(f"  ContextRelevance:    {scores['context_relevance']:.4f}")
    print("-" * 60)
    # 简单评级
    for name, score in scores.items():
        if score >= 0.80:
            bar = "[OK]"
        elif score >= 0.60:
            bar = "[WARN]"
        else:
            bar = "[LOW]"
        print(f"  {bar} {name}: {score:.4f}")
    print("=" * 60 + "\n")

    # 优化建议
    if scores["faithfulness"] < 0.70:
        print("  [!] Faithfulness 偏低 → LLM 可能编造了未在条款中出现的内容")
        print("      建议：检查 generate_node 的 prompt 中 context 是否正确注入")
    if scores["context_relevance"] < 0.60:
        print("  [!] ContextRelevance 偏低 → 检索召回了不相关的条款")
        print("      建议：收紧检索预检门控阈值（dense < 0.30 → 可尝试 0.35）")
    if scores["context_recall"] < 0.60:
        print("  [!] ContextRecall 偏低 → 检索漏召回了部分相关条款")
        print("      建议：检查 BM25 分词是否覆盖合同特有术语；增大 retrieval_top_k")
    print()


# ============================================================
# pytest 入口
# ============================================================
@pytest.mark.slow
@pytest.mark.eval
def test_ragas_evaluation():
    """
    Ragas 评估测试用例。

    运行方式：
        uv run pytest tests/eval/test_ragas.py -v -s -m eval

    注意：
        此测试依赖 Qdrant 中已有索引的合同数据。
        如无数据，先通过前端上传合同并等待 Worker 解析完成。
    """
    result = run_evaluation()
    scores = result["scores"]

    # 记录基线（不作为硬断言，避免因数据缺失导致 CI 失败）
    print(
        f"\n  评估完成: {result['samples']} 条样本, "
        f"Faithfulness={scores['faithfulness']:.4f}"
    )

    # 确保至少完成了评估
    assert result["samples"] >= 4, f"样本数不足: {result['samples']}"
    assert all(0.0 <= v <= 1.0 for v in scores.values()), f"分数值异常: {scores}"


# ============================================================
# 直接运行入口
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )
    run_evaluation()
