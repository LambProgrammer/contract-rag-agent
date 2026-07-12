"""
合同处理异步任务（process_contract）
====================================
在 RAG 系统中的角色：
    这是 M1 的核心任务编排器——将步骤 3（解析）、步骤 4（分块）、
    步骤 5（向量化 + 索引）串联为一条 Celery 异步流水线。

为什么需要异步（再次强调）：
    一份 29 页的合同 CPU 解析需要 2-3 分钟，不能阻塞 HTTP 请求。
    Celery Worker 在后台执行完整流水线，用户通过 task_id 轮询进度。

任务执行流程：
    process_contract(self, doc_id, file_path)
      │
      ├─ 10%   → 开始解析 (Docling / Unstructured 回退)
      ├─ 40%   → 解析完成，开始分块 (LegalClauseChunker)
      ├─ 60%   → 分块完成，加载 Embedding 模型
      ├─ 70%   → 向量化 Chunk 文本 (BAAI/bge-small-zh-v1.5)
      ├─ 85%   → 写入 Qdrant (批量 upsert)
      └─ 100%  → 完成，返回汇总信息

失败与重试：
    任务默认最多重试 3 次（见 celery_app.py 全局配置）。
    重试时从第一步重新开始（不依赖中间结果，保证幂等性）。

用法（由 upload.py 调用）：
    from src.tasks.process_contract import process_contract
    task = process_contract.delay(doc_id="abc", file_path="/app/uploads/abc/合同.pdf")
"""

import logging
import time
from pathlib import Path

from celery import Task, shared_task

from src.core.config import settings

logger = logging.getLogger(__name__)


# 使用 shared_task 而非 @app.task，避免与 celery_app.py 之间形成循环导入。
# shared_task 使用"当前 Celery 应用"（在 Worker 启动时自动绑定），
# celery_app.py 通过 `import src.tasks.process_contract` 触发装饰器注册。
@shared_task(
    bind=True,  # bind=True 让 self 指向当前任务实例，可调用 self.update_state()
    max_retries=3,  # 失败后最多重试 3 次
    name="process_contract",  # 显式命名（便于在 Celery Flower 等监控工具中识别）
    acks_late=True,  # 任务执行完成后才确认（Worker 崩溃时任务不丢失）
)
def process_contract(self: Task, doc_id: str, file_path: str) -> dict:
    """
    Celery 异步任务：处理上传的合同文件。

    参数：
        doc_id:    文档唯一 ID（与上传时返回的 doc_id 一致）
        file_path: 上传文件的绝对路径（容器内路径，如 /app/uploads/abc/合同.pdf）

    返回：
        {
            "doc_id": "abc",
            "chunk_count": 30,
            "parse_time_s": 82.3,
            "parser_used": "docling",
            "file_size_kb": 356,
        }
    """
    # ============================================================
    # 前置检查：文件是否存在
    # ============================================================
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(
            f"待解析文件不存在: {file_path}。文件可能已被删除或 volume 挂载不正确。"
        )

    file_size_kb = fp.stat().st_size / 1024
    logger.info(
        f"[Task] ========== 开始处理合同 ==========\n"
        f"       doc_id: {doc_id}\n"
        f"       文件:   {fp.name}\n"
        f"       大小:   {file_size_kb:.0f} KB"
    )

    total_start = time.perf_counter()

    # ============================================================
    # Phase 1: 解析文档（10% → 40%）
    # ============================================================
    self.update_state(state="PROCESSING", meta={"progress": 10, "stage": "解析文档"})

    from src.parsers import parse_document

    try:
        parse_result = parse_document(fp, doc_id)
    except RuntimeError as e:
        # ---- Hugging Face 网络错误的友好提示 ----
        error_msg = str(e)
        if "hf" in error_msg.lower() or "huggingface" in error_msg.lower():
            raise RuntimeError(
                f"文档解析失败——无法从 Hugging Face Hub 下载 AI 模型。\n"
                f"\n"
                f"解决方案：\n"
                f"  1. 确保容器能访问 https://huggingface.co\n"
                f"  2. 或设置环境变量 HF_ENDPOINT=https://hf-mirror.com\n"
                f"  3. 或将模型缓存目录挂载到容器中\n"
                f"\n"
                f"文件: {fp.name}\n"
                f"原始错误: {e}"
            ) from e
        raise

    parse_time_s = parse_result.parse_time_ms / 1000
    self.update_state(
        state="PROCESSING",
        meta={
            "progress": 40,
            "stage": "分块",
            "doc_id": doc_id,
            "parser_used": parse_result.parser_used,
            "page_count": parse_result.page_count,
        },
    )

    # ============================================================
    # Phase 2: 条款分块（40% → 60%）
    # ============================================================
    self.update_state(state="PROCESSING", meta={"progress": 50, "stage": "分块"})

    from src.chunkers.legal_chunker import LegalClauseChunker

    chunker = LegalClauseChunker()
    chunks = chunker.chunk(parse_result.text, doc_id)

    if not chunks:
        raise ValueError(
            f"分块结果为空——文档 {doc_id} ({fp.name}) 可能不包含可识别的条款文本。"
        )

    self.update_state(
        state="PROCESSING",
        meta={
            "progress": 60,
            "stage": "向量化",
            "doc_id": doc_id,
            "chunk_count": len(chunks),
        },
    )

    # ============================================================
    # Phase 3: 向量化（60% → 85%）
    # ============================================================
    self.update_state(
        state="PROCESSING", meta={"progress": 65, "stage": "加载 Embedding 模型"}
    )

    from src.indexing.embedder import BGEEmbedder

    embedder = BGEEmbedder()
    # 提取所有 Chunk 的文本
    chunk_texts = [c.text for c in chunks]

    self.update_state(
        state="PROCESSING",
        meta={"progress": 70, "stage": f"向量化 {len(chunk_texts)} 条文本"},
    )

    try:
        vectors = embedder.encode(chunk_texts)
    except RuntimeError as e:
        # ---- Hugging Face 网络错误的友好提示 ----
        raise RuntimeError(
            f"Embedding 模型加载失败——无法从 Hugging Face Hub 下载模型。\n"
            f"\n"
            f"Embedding 模型: {settings.embedding_model}\n"
            f"\n"
            f"解决方案：\n"
            f"  1. 确保容器能访问 https://huggingface.co\n"
            f"  2. 或设置环境变量 HF_ENDPOINT=https://hf-mirror.com\n"
            f"  3. 或预先下载模型到容器内的缓存目录\n"
            f"\n"
            f"原始错误: {e}"
        ) from e

    # ---- 生成稀疏向量（jieba 分词 → token_id → 词频权重）----
    self.update_state(
        state="PROCESSING", meta={"progress": 80, "stage": "生成稀疏向量"}
    )

    from src.retrieval.sparse_retriever import text_to_sparse_vector

    sparse_vectors = []
    for chunk in chunks:
        indices, values = text_to_sparse_vector(chunk.text)
        sparse_vectors.append((indices, values))

    self.update_state(
        state="PROCESSING",
        meta={"progress": 85, "stage": "写入 Qdrant"},
    )

    # ============================================================
    # Phase 4: 写入 Qdrant（85% → 95%）
    # ============================================================
    from src.indexing.vector_client import QdrantVectorClient

    qdrant = QdrantVectorClient()
    qdrant.ensure_collection()
    upserted_count = qdrant.upsert_chunks(
        chunks, vectors, sparse_vectors=sparse_vectors
    )

    total_time_s = time.perf_counter() - total_start

    # ============================================================
    # 完成
    # ============================================================
    result = {
        "doc_id": doc_id,
        "task_id": self.request.id,
        "chunk_count": len(chunks),
        "upserted_count": upserted_count,
        "parse_time_s": round(parse_time_s, 1),
        "total_time_s": round(total_time_s, 1),
        "parser_used": parse_result.parser_used,
        "page_count": parse_result.page_count,
        "file_size_kb": round(file_size_kb, 0),
        "file_name": fp.name,
    }

    logger.info(
        f"[Task] ========== 合同处理完成 ==========\n"
        f"       doc_id:   {doc_id}\n"
        f"       分块数:   {len(chunks)}\n"
        f"       向量数:   {upserted_count}\n"
        f"       解析耗时: {parse_time_s:.0f}s\n"
        f"       总耗时:   {total_time_s:.0f}s\n"
        f"       解析器:   {parse_result.parser_used}"
    )

    return result
