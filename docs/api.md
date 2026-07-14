# API 文档

> Base URL: `http://localhost:8000/api/v1`

---

## 1. 上传合同

```
POST /api/v1/upload
```

**请求**：`multipart/form-data`

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | file (required) | PDF / DOC / DOCX 合同文件，最大 10MB |

**响应**：`202 Accepted`

```json
{
  "doc_id": "a1b2c3d4-...",
  "task_id": "9e4f8a2c-...",
  "status": "pending",
  "message": "文件已接收，正在排队等待解析"
}
```

---

## 2. 查询任务进度

```
GET /api/v1/tasks/{task_id}
```

**响应**：`200 OK`

```json
{
  "task_id": "9e4f8a2c-...",
  "status": "completed",
  "progress": 100,
  "doc_id": "a1b2c3d4-...",
  "stage": "向量化",
  "result": {
    "chunk_count": 31,
    "parse_time_s": 22.1,
    "parser_used": "docling"
  }
}
```

状态枚举：`pending` → `processing` → `completed` / `failed`

---

## 3. 合同问答（SSE 流式）

```
POST /api/v1/qa
```

**请求**：`application/json`

```json
{
  "query": "旅行社在什么情况下需要退还全部费用？",
  "session_id": "可选，用于多轮对话",
  "doc_id": "可选，限定检索范围"
}
```

**响应**：`200 OK`，`text/event-stream`

```
event: token
data: {"content": "根"}

event: token
data: {"content": "据"}

...

event: done
data: {
  "session_id": "xxx",
  "rewritten_query": "旅行社退还全部费用的情形",
  "sources": [
    {
      "clause_ref": "第十六条 旅行社的违约责任",
      "text_snippet": "旅行社未按约定安排行程...",
      "score": 0.852
    }
  ]
}
```

---

## 4. 风险检测

```
POST /api/v1/risk/detect?doc_id={doc_id}
```

**请求**：无 body，通过 query string 传 `doc_id`

**响应**：`200 OK`（异步，立即返回 task_id）

```json
{
  "task_id": "abc-123-...",
  "doc_id": "a1b2c3d4-..."
}
```

通过 `GET /api/v1/tasks/{task_id}` 轮询结果。完成后 `result` 字段包含风险列表：

```json
{
  "risks": [
    {
      "rule_id": "R001",
      "rule_name": "违约金比例过高",
      "severity": "高",
      "clause_ref": "第十六条 旅行社的违约责任",
      "analysis": "合同约定违约金为总价的80%...",
      "suggestion": "建议调整为不超过实际损失的30%"
    }
  ],
  "total_risks": 3,
  "keyword_hits": 5,
  "false_alarms_filtered": 2
}
```

---

## 5. 合同比对

```
POST /api/v1/compare
```

**请求**：`application/json`

```json
{
  "doc_ids": ["doc-id-a", "doc-id-b"]
}
```

**响应**：`200 OK`（异步，立即返回 task_id）

```json
{
  "task_id": "def-456-..."
}
```

通过 `GET /api/v1/tasks/{task_id}` 轮询结果。完成后 `result` 包含比对统计和差异列表：

```json
{
  "stats": {
    "total_aligned": 18,
    "identical": 10,
    "with_differences": 6,
    "only_in_a": 2,
    "only_in_b": 1
  },
  "differences": [
    {
      "clause_ref_a": "第十六条",
      "clause_ref_b": "第十六条",
      "analysis": "违约金比例从'协商确定'变更为'80%'..."
    }
  ]
}
```

---

## 7. 健康检查

```
GET /health
```

```json
{"status": "ok", "app": "Contract RAG Agent", "version": "0.1.0"}
```

```
GET /health/ready
```

```json
{
  "status": "ready",
  "checks": {"postgres": "pending", "redis": "pending", "qdrant": "pending"}
}
```

---

## Swagger UI

开发调试时可直接访问：http://localhost:8000/docs
