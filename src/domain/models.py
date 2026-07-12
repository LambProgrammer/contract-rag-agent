"""
数据模型（Domain Models）
========================
在 RAG 系统中的角色：
    定义系统中各模块之间传递的数据结构（DTO / Data Transfer Object）。
    FastAPI 利用这些 Pydantic 模型自动生成：
        - 请求参数校验（不合法的参数立即拒绝，415/422）
        - Swagger 文档中的请求/响应示例
        - JSON Schema（前端或外部系统可直接引用）

为什么集中放置而非分散在各路由文件：
    - 避免循环引用：upload.py 和 qa.py 可能互相引用对方的模型
    - 统一修改：字段名或类型变更只需改一处
    - IDE 友好：from src.domain.models import ... 一目了然

用法：
    from src.domain.models import UploadResponse, TaskStatusResponse
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 任务状态枚举
# ============================================================
class TaskStatus(str, Enum):
    """Celery 任务的四种生命周期状态"""

    PENDING = "pending"  # 任务已提交，等待 Worker 取走
    PROCESSING = "processing"  # Worker 正在执行（解析/分块/索引）
    COMPLETED = "completed"  # 全部流程成功完成
    FAILED = "failed"  # 任何步骤报错（解析失败/Embedding 失败等）


# ============================================================
# 上传请求 / 响应
# ============================================================
class UploadResponse(BaseModel):
    """上传合同文件后立即返回的内容。

    此时文件已保存到磁盘，异步任务已提交，
    但解析尚未开始——用户需要用 task_id 轮询进度。
    """

    doc_id: str = Field(
        ...,
        description="文档唯一ID（UUID4），后续可按此ID查询该合同的所有条款",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )
    task_id: str = Field(
        ...,
        description="Celery 任务ID（UUID4），用于查询异步解析进度",
        examples=["9e4f8a2c-1b3d-4e5f-6789-abcdef012345"],
    )
    status: str = Field(
        default="pending",
        description="任务初始状态固定为 'pending'",
    )
    message: str = Field(
        default="文件已接收，正在排队等待解析",
        description="给用户的可读提示",
    )


# ============================================================
# 任务状态查询
# ============================================================
class TaskStatusResponse(BaseModel):
    """GET /tasks/{task_id} 的响应。

    用户通过轮询此接口获取异步解析的进度。
    """

    task_id: str = Field(
        ...,
        description="Celery 任务ID",
    )
    status: TaskStatus = Field(
        ...,
        description="任务当前状态：pending / processing / completed / failed",
    )
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="任务进度百分比（0-100），仅在 processing 时有意义",
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="关联的文档ID，任务完成后可用来查询条款",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="失败时的错误描述，仅 status=failed 时有值",
    )
    stage: Optional[str] = Field(
        default=None,
        description="当前处理阶段描述（如'解析文档''分块''向量化'），仅 processing 时有值",
    )
    result: Optional[dict] = Field(
        default=None,
        description="任务完成后的完整返回数据（仅 status=completed 时有值）",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="任务创建时间（容器本地时间）",
    )


# ============================================================
# QA 问答请求 / 响应
# ============================================================
class SourceDoc(BaseModel):
    """检索召回的单个合同条款来源。

    每个 QA 答案都附带多个 SourceDoc，让用户能追溯 LLM 答案的出处。
    这是 RAG 区别于普通聊天机器人的核心特征——答案必须有来源可查。
    """

    clause_ref: str = Field(
        ...,
        description="条款引用标识，如'第十六条 旅行社的违约责任'",
    )
    text_snippet: str = Field(
        ...,
        description="条款原文片段（截取前 200 字符），供用户快速确认相关性",
    )
    score: float = Field(
        ...,
        description="检索相似度得分（Cosine），反映该条款与提问的语义匹配程度",
    )


class QARequest(BaseModel):
    """用户提问请求。"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户的自然语言提问，如'违约金怎么计算？'",
        examples=["旅行社在什么情况下需要退还全部费用？"],
    )
    session_id: str | None = Field(
        default=None,
        description="可选的会话 ID（UUID）。不传则自动生成新的。"
        "同一 session_id 的多次请求共享对话历史",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    doc_id: str | None = Field(
        default=None,
        description="可选的文档 ID。传入后检索仅限该合同范围，避免多合同混排。"
        "不传则检索全部已入库合同",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


class QAResponse(BaseModel):
    """QA 端点的完整响应：LLM 答案 + 引用的合同条款来源。"""

    session_id: str = Field(
        ...,
        description="当前会话 ID。后续提问携带同一 ID 即可共享对话历史",
    )
    query: str = Field(
        ...,
        description="用户原始提问（原样返回，方便前端展示）",
    )
    answer: str = Field(
        ...,
        description="LLM 基于合同条款生成的答案文本",
    )
    rewritten_query: str = Field(
        default="",
        description="改写后的查询（如有），用于调试检索质量",
    )
    sources: list[SourceDoc] = Field(
        default_factory=list,
        description="检索召回的合同条款来源列表，每项包含条款引用和原文片段",
    )
