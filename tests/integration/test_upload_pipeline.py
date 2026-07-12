"""
集成测试：上传→解析→QA 全链路
需要 Docker Desktop（postgres + redis + qdrant）+ Worker 运行
运行方式：uv run pytest tests/integration/ -v -m integration
"""

import uuid

import pytest
import requests

BASE = "http://localhost:8000/api/v1"


@pytest.mark.integration
@pytest.mark.slow
class TestUploadPipeline:
    """测试上传 PDF → 等待 Worker 解析 → QA 提问的完整链路"""

    def test_upload_and_qa(self):
        # 上传 PDF
        import os

        pdf_path = os.path.join(
            os.path.dirname(__file__), "..", "fixtures", "sample.pdf"
        )
        if not os.path.exists(pdf_path):
            pytest.skip("测试 PDF 文件不存在，跳过集成测试")

        with open(pdf_path, "rb") as f:
            r = requests.post(
                f"{BASE}/upload",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )
        assert r.status_code == 202
        doc_id = r.json()["doc_id"]
        task_id = r.json()["task_id"]

        # 轮询任务进度
        import time

        for _ in range(60):  # 最多等 2 分钟
            tr = requests.get(f"{BASE}/tasks/{task_id}")
            if tr.json()["status"] == "completed":
                break
            if tr.json()["status"] == "failed":
                pytest.fail(f"任务失败: {tr.json().get('error_message')}")
            time.sleep(2)
        else:
            pytest.fail("任务在 2 分钟内未完成")

        # QA 提问
        sid = str(uuid.uuid4())
        qr = requests.post(
            f"{BASE}/qa",
            json={"query": "违约责任怎么算？", "session_id": sid, "doc_id": doc_id},
        )
        assert qr.status_code == 200
