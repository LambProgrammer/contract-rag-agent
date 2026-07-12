"""
集成测试：上传→风险检测 全链路
需要 Docker Desktop + Worker 运行
运行方式：uv run pytest tests/integration/ -v -m integration
"""

import pytest
import requests

BASE = "http://localhost:8000/api/v1"


@pytest.mark.integration
@pytest.mark.slow
class TestRiskPipeline:
    """测试上传 PDF → 等待 Worker 解析 → 风险检测"""

    def test_risk_detection(self):
        import os
        import time

        pdf_path = os.path.join(
            os.path.dirname(__file__), "..", "fixtures", "sample.pdf"
        )
        if not os.path.exists(pdf_path):
            pytest.skip("测试 PDF 文件不存在，跳过集成测试")

        # 上传并等待解析完成
        with open(pdf_path, "rb") as f:
            r = requests.post(
                f"{BASE}/upload",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )
        task_id = r.json()["task_id"]
        doc_id = r.json()["doc_id"]

        for _ in range(60):
            tr = requests.get(f"{BASE}/tasks/{task_id}")
            if tr.json()["status"] == "completed":
                break
            time.sleep(2)

        # 风险检测（异步）
        rr = requests.post(f"{BASE}/risk/detect?doc_id={doc_id}")
        assert rr.status_code == 200
        risk_task_id = rr.json().get("task_id")
        assert risk_task_id is not None

        # 等待风险检测完成
        for _ in range(60):
            tr = requests.get(f"{BASE}/tasks/{risk_task_id}")
            if tr.json()["status"] == "completed":
                break
            time.sleep(2)

        result = requests.get(f"{BASE}/tasks/{risk_task_id}").json()
        assert result["status"] == "completed"
        assert result.get("result") is not None
