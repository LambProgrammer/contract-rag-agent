"""
Embedding 向量化模块
===================
在 RAG 系统中的角色：
    将步骤 4 的 Chunk 文本转换为稠密向量（embedding），
    这是"文本 → 数学向量"的转换器，让计算机能"理解"文本的语义。

什么是 Embedding（学习参考）：
    想象一个 512 维的空间，每个 Chunk 文本都是这个空间中的一个点。
    语义相似的文本（如"违约责任"和"违约金条款"）在这个空间中距离很近，
    语义无关的文本（如"违约责任"和"付款方式"）距离很远。
    这就是 RAG 检索的原理——用户提问被转为向量后，
    在 Qdrant 中找"距离最近的" Chunk 向量，返回对应的原文。

为什么选择 BAAI/bge-small-zh-v1.5：
    - 中文优化：BAAI（北京智源研究院）出品，专为中文语义设计
    - 体积小：102M 参数，约 400MB 磁盘，Worker 内存友好
    - 效果好：在中文法律文本语义相似度基准上表现优异
    - 本地运行：不需要调 API，无网络延迟，无调用费用

向量维度 512 的含义：
    每个 Chunk 文本被压缩为 512 个浮点数 [0.12, -0.34, 0.89, ...]。
    这 512 个数字就是文本的"语义指纹"。
    512 维是 bge-small 的标准输出维度，不需要在 .env 中额外配置。

用法：
    from src.indexing.embedder import BGEEmbedder
    embedder = BGEEmbedder()
    vectors = embedder.encode(["条款文本1", "条款文本2"])
    # → numpy.ndarray 形状 (2, 512)
"""

import logging
import time
from typing import List

import numpy as np
from langfuse import observe

from src.core.config import settings

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """
    BAAI/bge-small-zh-v1.5 Embedding 模型封装。

    使用 sentence-transformers 库加载模型，提供批量文本向量化接口。
    模型在首次调用 encode() 时延迟加载，避免 Worker 启动时就占用内存。
    """

    def __init__(self) -> None:
        self._model = None  # 延迟加载
        self._dim = 512  # bge-small-zh-v1.5 固定输出 512 维

    @property
    def model(self):
        """
        延迟加载 sentence-transformers 模型。

        为什么延迟加载：
            Worker 启动时如果立即加载模型，每个 Worker 进程都占用 ~400MB 内存。
            延迟加载意味着只有真正执行解析任务时才加载，空闲 Worker 不占内存。

        网络错误提示：
            首次加载时需要从 Hugging Face Hub 下载模型文件（~400MB）。
            如果网络不通，会给出明确的中文提示和解决方案。
        """
        if self._model is None:
            logger.info(f"[Embedder] 正在加载模型: {settings.embedding_model}...")
            load_start = time.perf_counter()

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers 未安装。请运行: uv add sentence-transformers"
                )

            # ----------------------------------------------------------
            # 加载策略：优先本地缓存，本地没有时才联网下载
            #
            # local_files_only=True:
            #   不发起任何网络请求，直接从本地 HF 缓存读取。加载速度快（<1秒），
            #   不需要科学上网。适用于模型已下载到本地的场景（绝大多数情况）。
            #
            # local_files_only=False:
            #   首次使用或更新模型时启用。会先联网检查版本（HEAD 请求），
            #   如果国内网络不通会导致超时。仅在 True 模式失败时作为回退。
            # ----------------------------------------------------------
            try:
                logger.info("[Embedder] 尝试从本地缓存加载（不联网）...")
                self._model = SentenceTransformer(
                    settings.embedding_model,
                    device="cpu",
                    local_files_only=True,  # 仅使用本地缓存，不联网检查更新
                )
                logger.info("[Embedder] 模型从本地缓存加载成功")
            except Exception:
                # 本地缓存没有 → 回退到联网下载
                logger.info("[Embedder] 本地缓存未找到模型，尝试从 HuggingFace 下载...")
                try:
                    self._model = SentenceTransformer(
                        settings.embedding_model,
                        device="cpu",
                        local_files_only=False,
                    )
                except OSError as e:
                    error_msg = str(e).lower()
                    if any(
                        kw in error_msg
                        for kw in [
                            "connection",
                            "timeout",
                            "hub",
                            "huggingface",
                            "resolve",
                            "refused",
                        ]
                    ):
                        raise RuntimeError(
                            f"无法加载 Embedding 模型 "
                            f"'{settings.embedding_model}'。\n"
                            f"\n"
                            f"模型既不在本地缓存中，也无法从 HuggingFace 下载。\n"
                            f"\n"
                            f"解决方案：\n"
                            f"  1. 一次性下载 → 在有网络的条件下运行一次，模型会被缓存到本地\n"
                            f"  2. 国内镜像 → 设置环境变量 HF_ENDPOINT=https://hf-mirror.com\n"
                            f"  3. 手动下载 → "
                            f"将模型文件放入 '{settings.embedding_model}' 对应目录\n"
                            f"\n"
                            f"原始错误: {e}"
                        ) from e
                    raise

            load_time = (time.perf_counter() - load_start) * 1000
            logger.info(f"[Embedder] 模型加载完成，耗时 {load_time:.0f}ms")

        return self._model

    @observe(name="embed-documents", as_type="embedding")
    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        将文本列表批量转为向量。

        参数：
            texts: 待向量化的文本列表（通常是 Chunk.text 列表）
            show_progress: 是否显示进度条（CLI 调试时可用）

        返回：
            numpy.ndarray，形状 (len(texts), 512)，dtype=float32

        示例：
            embedder = BGEEmbedder()
            vectors = embedder.encode(["第一条 违约责任", "第二条 费用支付"])
            assert vectors.shape == (2, 512)
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        logger.info(f"[Embedder] 开始向量化 {len(texts)} 段文本...")
        start = time.perf_counter()

        # ---- 核心 API：SentenceTransformer.encode() ----
        # normalize_embeddings=True:
        #   将每条向量的模长归一化为 1，使余弦相似度计算变成简单的向量内积，
        #   这是 Qdrant + Cosine 距离的推荐配置
        # batch_size:
        #   每次推理处理 32 条文本，平衡速度和内存
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=32,
        )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"[Embedder] 向量化完成: {len(texts)} 条, "
            f"{elapsed:.0f}ms ({elapsed / len(texts):.0f}ms/条)"
        )

        return embeddings  # type: ignore[reportReturnType]

    @property
    def dimension(self) -> int:
        """向量维度（512）"""
        return self._dim
