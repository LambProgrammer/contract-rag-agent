"""
合同比对模块（Comparator）
==========================
在 RAG 系统中的角色：
    对两份已入库合同进行自动化条款级对比，输出差异报告。

模块结构：
    diff_analyzer.py  — B0 相关性预检 + B1 条款对齐 + B2 预过滤 + B3 LLM 对比
    router.py         — B4 比对 API 端点 + 结果 Redis 暂存
"""
