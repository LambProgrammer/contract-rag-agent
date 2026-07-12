"""
Qdrant 向量数据库客户端
=======================
在 RAG 系统中的角色：
    将步骤 4 的 Chunk 文本和步骤 5 的 Embedding 向量存入 Qdrant，
    后续 M2 的检索操作也通过这个客户端查询。

Qdrant 中的数据结构（学习参考）：
    集合（Collection）  ≈ 关系数据库中的"表"
    Point               ≈ 表中的"一行"
        id              → 主键（我们用 chunk_id）
        vector          → BGE 模型输出的 512 维向量
        payload         → 附加元数据（JSON 格式）
            - doc_id      来源文档
            - chunk_index 原文中的位置
            - clause_ref  条款引用（"第五条"）
            - text        原始文本（检索时直接返回，无需回查 DB）

为什么在写入时同时保存原始文本（text in payload）：
    RAG 检索的标准流程是"向量检索 → 返回原文"。
    如果 payload 里不存 text，检索后还需要回 PostgreSQL 查原文，
    多一次网络调用。存 payload 中则一次 Qdrant 查询即可返回完整结果。

集合配置说明：
    - 向量维度: 512（BAAI/bge-small-zh-v1.5 固定输出）
    - 距离度量: Cosine（余弦相似度，适合归一化向量）
    - 当前阶段只做稠密向量索引，稀疏向量（BM25）在 M3 阶段添加

用法：
    from src.indexing.vector_client import QdrantVectorClient
    client = QdrantVectorClient()
    client.ensure_collection()
    client.upsert_chunks(chunks, vectors)
"""

import logging
import uuid
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import settings
from src.chunkers.legal_chunker import Chunk

logger = logging.getLogger(__name__)


class QdrantVectorClient:
    """
    Qdrant 向量数据库操作封装。

    提供集合管理、向量写入、删除等操作。
    内部使用 qdrant-client 的 HTTP API（不依赖 gRPC）。
    """

    def __init__(self) -> None:
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            # 使用 HTTP 协议（而非 gRPC），更简单、兼容性更好
            prefer_grpc=False,
        )
        self._collection_name = settings.qdrant_collection_name

    # ----------------------------------------------------------
    # 集合管理
    # ----------------------------------------------------------
    def ensure_collection(self) -> None:
        """
        确保 Qdrant 中存在目标集合，不存在则创建。

        幂等操作：多次调用不会重复创建或报错。

        集合配置：
            - 向量维度 512（bge-small-zh-v1.5 输出维度）
            - Cosine 距离（向量已归一化，Cosine 退化为内积）
            - on_disk_payload=True（payload 存磁盘，减少内存占用）
        """
        # 检查集合是否已存在
        existing_cols = self._client.get_collections()
        for col in existing_cols.collections:
            if col.name == self._collection_name:
                # M3: 旧集合可能只有 dense vector，需重建以添加 sparse vector
                info = self._client.get_collection(self._collection_name)
                has_sparse = bool(
                    info.config.params.sparse_vectors
                    if info.config and info.config.params.sparse_vectors
                    else False
                )
                if not has_sparse:
                    logger.info(
                        f"[Qdrant] 集合 '{self._collection_name}' 缺少稀疏向量配置，"
                        f"正在删除并重建..."
                    )
                    self._client.delete_collection(self._collection_name)
                    break  # 退出循环，执行下面的创建逻辑
                logger.info(
                    f"[Qdrant] 集合 '{self._collection_name}' 已存在（含 dense + sparse），跳过创建"
                )
                return

        # 创建新集合（含 dense + sparse 两种向量）
        logger.info(
            f"[Qdrant] 正在创建集合 '{self._collection_name}' "
            f"（向量维度: 512 dense + sparse, 距离: Cosine）..."
        )
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                "dense": qmodels.VectorParams(
                    size=512,
                    distance=qmodels.Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(),
            },
        )
        logger.info(f"[Qdrant] 集合 '{self._collection_name}' 创建完成")

    # ----------------------------------------------------------
    # 写入操作
    # ----------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: List[Chunk],
        vectors: "np.ndarray",
        sparse_vectors: Optional[List[tuple]] = None,
    ) -> int:
        """
        批量写入 Chunk 及其向量到 Qdrant。

        参数：
            chunks:         分块列表（每个 Chunk 的 text/clause_ref 存入 payload）
            vectors:        numpy 数组，形状 (len(chunks), 512)，稠密向量
            sparse_vectors: 可选，每个 Chunk 的稀疏向量 [(indices, values), ...]，
                           indices 和 values 都是 list。用于 BM25 关键词检索。

        返回：
            写入的 Point 数量
        """
        import numpy as np

        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks 和 vectors 数量不一致: {len(chunks)} vs {len(vectors)}"
            )
        if sparse_vectors and len(sparse_vectors) != len(chunks):
            raise ValueError(
                f"sparse_vectors 和 chunks 数量不一致: "
                f"{len(sparse_vectors)} vs {len(chunks)}"
            )

        if len(chunks) == 0:
            return 0

        # 构建 Qdrant Point 列表
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # 确保 chunk_id 是合法的 UUID（Qdrant 要求）
            point_id = str(chunk.chunk_id)
            try:
                uuid.UUID(point_id)
            except ValueError:
                point_id = str(uuid.uuid4())

            # 将 numpy 向量转为 Python list（Qdrant HTTP API 要求）
            vector_list = (
                vector.tolist() if isinstance(vector, np.ndarray) else list(vector)
            )

            point = qmodels.PointStruct(
                id=point_id,
                vector={
                    "dense": vector_list,
                },
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "clause_ref": chunk.clause_ref,
                    "text": chunk.text,
                },
            )

            # ---- 附加稀疏向量 ----
            if sparse_vectors and i < len(sparse_vectors):
                sp_indices, sp_values = sparse_vectors[i]
                if sp_indices and sp_values:
                    point.vector["sparse"] = qmodels.SparseVector(
                        indices=sp_indices,
                        values=sp_values,
                    )

            points.append(point)

        # ---- 批量写入 ----
        # wait=true: 等待写入完成再返回，保证后续检索能查到
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

        logger.info(
            f"[Qdrant] 写入完成: {len(points)} 个 Point → "
            f"集合 '{self._collection_name}'"
        )
        return len(points)

    # ----------------------------------------------------------
    # 查询与删除（M2 阶段使用，此处预留给入口）
    # ----------------------------------------------------------
    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        删除指定文档的所有向量 Point。

        使用场景：用户重新上传同份合同 → 删除旧索引 → 重新索引。
        Qdrant 支持按 payload 字段过滤删除，不需要遍历所有 Point。
        """
        result = self._client.delete(
            collection_name=self._collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="doc_id",
                            match=qmodels.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        deleted_count = (
            getattr(result.status, "deleted", 0) if hasattr(result, "status") else 0
        )
        logger.info(f"[Qdrant] 删除文档 {doc_id}: {deleted_count} 个 Point")
        return deleted_count

    def search(
        self,
        query_vector: "np.ndarray",
        top_k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> List:
        """
        向量检索（M2 使用）。

        参数：
            query_vector: 查询向量（512 维）
            top_k: 返回 Top-K 结果
            doc_id_filter: 可选，限定在特定文档内检索

        返回：
            匹配的 scored points 列表
        """
        query_filter = None
        if doc_id_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=doc_id_filter),
                    )
                ]
            )

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=qmodels.NamedVector(
                name="dense",
                vector=query_vector.tolist()
                if hasattr(query_vector, "tolist")
                else query_vector,
            ),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return results

    def search_sparse(
        self,
        indices: List[int],
        values: List[float],
        top_k: int = 5,
    ) -> List:
        """
        稀疏向量检索（M3 新增 — BM25 关键词匹配）。

        参数：
            indices: 稀疏向量的词 ID 列表
            values:  对应的词权重列表
            top_k:   返回 Top-K 结果

        返回：
            匹配的 scored points 列表
        """
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=qmodels.NamedSparseVector(
                name="sparse",
                vector=qmodels.SparseVector(
                    indices=indices,
                    values=values,
                ),
            ),
            limit=top_k,
            with_payload=True,
        )
        return results

    def collection_info(self) -> dict:
        """获取集合统计信息（调试用）"""
        info = self._client.get_collection(self._collection_name)
        return {
            "name": self._collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
        }
