"""
文件上传 API
============
在 RAG 系统中的角色：
    连接"用户"与"异步解析管道"的桥梁。
    收到文件后只做最轻量的工作（验证 + 保存 + 提交任务），
    立即返回 task_id，不阻塞 HTTP 连接。

与 Celery 的协作方式：
    1. 用户 POST /api/v1/upload 上传文件
    2. 本模块验证格式、保存文件、调用 process_contract.delay(doc_id, file_path)
    3. Celery Worker 在后台取走任务，执行解析→分块→索引
    4. 用户 GET /api/v1/tasks/{task_id} 轮询进度

为什么文件保存在本地而非 S3/OSS：
    - M1 学习阶段优先简单可调试，uploads/ 通过 Docker volume 持久化
    - 后续可替换为对象存储（只需改文件读写部分，不影响其他模块）

用法：
    在 src/api/main.py 中注册：
    from src.api.upload import router
    app.include_router(router)
"""

import os
import uuid
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.core.config import settings
from src.domain.models import TaskStatus, TaskStatusResponse, UploadResponse
from src.tasks.celery_app import app as celery_app

# ============================================================
# 常量定义
# ============================================================
# 允许上传的文件 MIME 类型
# PDF: PDF 合同（最常见）
# DOCX: Word 合同 (.docx)
# DOC: 旧版 Word 合同 (.doc)
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",  # .doc
}

# 允许的文件扩展名（双重校验，防 MIME 伪装）
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}

router = APIRouter(prefix=settings.api_v1_prefix, tags=["上传与任务"])


# ============================================================
# 辅助函数
# ============================================================
def _validate_file(file: UploadFile) -> None:
    """
    验证上传文件的格式和大小。

    为什么做双重校验（MIME + 扩展名）：
        MIME type 来自浏览器声明，可以被伪造（例如恶意文件声明自己是 PDF），
        扩展名检查作为第二道防线，但扩展名也可以伪造。
        真正的格式识别需要解析文件头（magic bytes），
        但 Docling/Unstructured 在步骤 3 解析时会再次验证并报错。
        这里只是粗粒度的快速拦截。

    抛出 HTTPException 时状态码的选择：
        413 文件过大 — 用户不需要重试当前文件
        415 格式不支持 — 提示用户转换格式后再试
    """
    # ---- 检查文件大小 ----
    # UploadFile.size 来自请求头的 Content-Length，不读文件内容，性能无影响
    if file.content_type is None:
        raise HTTPException(
            status_code=415,
            detail="无法识别文件类型，请检查文件格式",
        )

    # ---- 检查 MIME 类型 ----
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型 '{file.content_type}'。"
            f"请上传 PDF 或 Word (.docx/.doc) 格式的合同文件",
        )

    # ---- 检查扩展名 ----
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"不支持的文件扩展名 '{ext}'。"
                f"请上传 .pdf / .docx / .doc 格式的合同文件",
            )
    # ---- 检查文件大小 ----
    # 读入文件内容以获取真实大小（UploadFile.size 来自 Content-Length 头，可能不可靠）
    file.file.seek(0)
    content = file.file.read()
    file.file.seek(0)  # 重置指针，后续 _save_uploaded_file 可重新读取
    if len(content) > settings.max_upload_size:
        size_mb = len(content) / (1024 * 1024)
        limit_mb = settings.max_upload_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"文件过大：{size_mb:.1f}MB，超过限制 {limit_mb:.0f}MB",
        )


def _save_uploaded_file(file: UploadFile, doc_id: str) -> Path:
    """
    将上传的文件保存到本地文件系统。

    目录结构：uploads/{doc_id}/{原始文件名}
    例如：  uploads/a1b2c3d4.../三方合同v3.pdf

    为什么用 doc_id 做子目录：
        - 同一份合同可能有多个版本（用户重复上传），方便追溯
        - 解析后的临时文件（图片、中间 JSON）也放这里，清理时一起删除
    """
    upload_dir = Path(settings.upload_dir) / doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 保留原始文件名（方便人工查看和调试）
    safe_filename = file.filename or "untitled"
    file_path = upload_dir / safe_filename

    # 分块写入（大文件不会撑爆内存）
    with open(file_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):  # 每次读 1MB
            f.write(chunk)

    return file_path


# ============================================================
# API 端点
# ============================================================
@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_contract(file: UploadFile = File(...)):
    """
    上传合同文件并提交异步解析任务。

    处理流程：
        1. 验证文件格式（MIME + 扩展名）
        2. 生成 doc_id（UUID4）
        3. 保存文件到 uploads/{doc_id}/ 目录
        4. 向 Celery 提交异步解析任务
        5. 立即返回 doc_id + task_id

    HTTP 202 Accepted：
        表示请求已接受但尚未处理完毕，客户端应轮询 GET /tasks/{task_id}。

    当前限制：
        - 解析任务尚未实现（步骤 5 完成），task_id 存在但任务会一直 pending
        - 文件大小未严格校验（M5 补充）
    """
    # ① 验证文件
    _validate_file(file)

    # ② 生成唯一文档 ID
    doc_id = str(uuid.uuid4())

    # ③ 保存文件
    file_path = _save_uploaded_file(file, doc_id)

    # ④ 提交异步任务
    from src.tasks.process_contract import process_contract

    task = process_contract.delay(  # type: ignore[reportCallIssue]
        doc_id=str(doc_id), file_path=file_path.as_posix()
    )
    task_id = task.id

    # ⑤ 返回响应
    return UploadResponse(
        doc_id=doc_id,
        task_id=task_id,
        status="pending",
        message=f"文件 '{file.filename}' 已接收（{os.path.getsize(file_path)} 字节），"
        f"排队等待解析中",
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
)
async def get_task_status(task_id: str):
    """
    查询异步解析任务的进度。

    用户上传后轮询此端点（建议间隔 2-3 秒），前端根据 status 决定：
        - pending / processing → 继续轮询
        - completed → 跳转到条款查看页
        - failed → 显示 error_message，引导用户重试

    工作原理：
        通过 Celery 的 AsyncResult 从 Redis 查询任务状态。
        因为 Redis 是内存存储，查询很快（<1ms），不需要缓存。
        Redis 中的任务结果默认 1 小时后过期（见 celery_app.py 配置）。
    """
    # Celery AsyncResult — 从 Redis result backend 查询任务元数据
    # app 必须显式传入 celery_app 实例，否则 Celery 找不到 backend 会报 DisabledBackend 错误
    task = AsyncResult(task_id, app=celery_app)

    # 将 Celery 的状态映射到我们的 TaskStatus 枚举
    status_map = {
        "PENDING": TaskStatus.PENDING,
        "STARTED": TaskStatus.PROCESSING,
        "RETRY": TaskStatus.PROCESSING,
        "SUCCESS": TaskStatus.COMPLETED,
        "FAILURE": TaskStatus.FAILED,
    }
    status = status_map.get(task.state, TaskStatus.PENDING)

    # 构建响应
    response = TaskStatusResponse(
        task_id=task_id,
        status=status,
        progress=_estimate_progress(task),
    )

    # 提取阶段描述（处理中时 task.info 里有 meta）
    if isinstance(task.info, dict):
        response.stage = task.info.get("stage")
        response.doc_id = task.info.get("doc_id")

    # 附加额外信息（如果任务已完成或失败）
    if task.ready():
        if isinstance(task.result, dict):
            response.doc_id = task.result.get("doc_id")
            response.error_message = task.result.get("error")
            response.result = (
                task.result
            )  # 完整返回数据（risk/compare 等异步任务的结果）
        elif isinstance(task.info, dict):
            response.progress = task.info.get("progress", 100)
            response.doc_id = task.info.get("doc_id")

    return response


def _estimate_progress(task: AsyncResult) -> int:
    """
    估算任务进度（百分比）。

    Celery 本身没有内置进度条，进度信息由 Worker 在执行过程中
    通过 task.update_state(state='PROGRESS', meta={'progress': 50}) 写入 Redis。

    当前 mock 阶段返回固定值，步骤 5 后由 Worker 更新。
    """
    if task.state == "SUCCESS":
        return 100
    if task.state == "FAILURE":
        return 0
    if isinstance(task.info, dict):
        return task.info.get("progress", 0)
    return 0
