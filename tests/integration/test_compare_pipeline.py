"""
集成测试：双合同比对 全链路
需要 Docker Desktop + Worker 运行
运行方式：uv run pytest tests/integration/ -v -m integration
"""

import pytest
import requests

BASE = "http://localhost:8000/api/v1"


@pytest.mark.integration
@pytest.mark.slow
class TestComparePipeline:
    """测试上传两份合同→等待解析→比对"""

    def test_compare_two_contracts(self):
        import os
        import time

        pdf_path = os.path.join(
            os.path.dirname(__file__), "..", "fixtures", "sample.pdf"
        )
        if not os.path.exists(pdf_path):
            pytest.skip("测试 PDF 文件不存在，跳过集成测试")

        # 上传第一份
        with open(pdf_path, "rb") as f:
            r1 = requests.post(
                f"{BASE}/upload",
                files={"file": ("sample_a.pdf", f, "application/pdf")},
            )
        task_a = r1.json()["task_id"]
        doc_a = r1.json()["doc_id"]

        # 上传第二份（同一文件模拟"修改版"）
        with open(pdf_path, "rb") as f:
            r2 = requests.post(
                f"{BASE}/upload",
                files={"file": ("sample_b.pdf", f, "application/pdf")},
            )
        task_b = r2.json()["task_id"]
        doc_b = r2.json()["doc_id"]

        # 等两份都解析完
        for tid in [task_a, task_b]:
            for _ in range(60):
                tr = requests.get(f"{BASE}/tasks/{tid}")
                if tr.json()["status"] == "completed":
                    break
                time.sleep(2)

        # 比对
        cr = requests.post(f"{BASE}/compare", json={"doc_ids": [doc_a, doc_b]})
        assert cr.status_code == 200
        compare_task_id = cr.json().get("task_id")
        assert compare_task_id is not None

        # 等待比对完成
        for _ in range(60):
            tr = requests.get(f"{BASE}/tasks/{compare_task_id}")
            if tr.json()["status"] == "completed":
                break
            time.sleep(2)

        result = requests.get(f"{BASE}/tasks/{compare_task_id}").json()
        assert result["status"] == "completed"
        assert result.get("result") is not None
