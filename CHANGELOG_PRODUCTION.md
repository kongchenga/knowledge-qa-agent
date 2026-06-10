# 生产化改造变更记录

> 改造日期: 2026-06-08
> 改造范围: 3 轮、19 个文件、25 项改进

---

## 改造概览

| 轮次 | 范围 | 文件数 |
|------|------|--------|
| 第一轮 | 性能 + 架构 (P0-P2) | 5 |
| 第二轮 | 安全 + 可靠性 (生产化) | 6 |
| 第三轮 | 细节打磨 (健壮性) | 7 |

---

## 第一轮: 性能与架构升级

### 1. 模型冷启动预热

**问题**: embedding 和 reranker 模型首次调用才懒加载，第一个请求阻塞 8-15 秒。

**文件**: `app/main.py`

**改动**: lifespan 中预加载并记录耗时:
```python
from app.embeddings import get_embedding_service
get_embedding_service().embed_query("warmup")
from app.reranker import get_reranker
get_reranker().rerank("warmup", [...])
```

**测试方法**: 启动后立即发 `POST /api/query`，第一个请求延迟应 <3s（不含 LLM 调用）。

---

### 2. BM25 索引增量存储

**问题**: 每次 add/delete 文档时 `bm25.json` 全量序列化写入磁盘。1000+ chunks 时每次写入 500KB+ JSON，I/O 瓶颈。

**文件**: `app/database/vector_store.py`

**改动**:
- `BM25Index.__init__()`: 新增 `_init_sqlite()` 方法，使用 SQLite 表 `docs` + `posting` 做增量存储
- `add_texts()`: 每新增一个 chunk 直接 `INSERT` 到 SQLite + 内存倒排索引，不再写 JSON
- `delete_by_doc_id()`: SQLite `DELETE` + 内存清理
- `_load()`: 从 `posting` 表批量重建倒排索引（O(1) 取代 O(n)），含从旧 JSON 格式的自动迁移逻辑
- `_save()` 方法移除

**测试方法**:
1. 上传 5 个 PDF 文档（约 50 chunks）
2. 检查 `data/chroma/bm25/bm25.json` 是否已被删除
3. 检查 `data/chroma/bm25/bm25.db` 是否存在
4. `SELECT COUNT(*) FROM docs` 应匹配 chunk 数

---

### 3. 消除 query/stream_query 重复代码

**问题**: `query()` 和 `stream_query()` 中有 30 行一模一样的 context 构建代码。

**文件**: `app/agent/qa_agent.py`

**改动**: 提取公共方法 `_build_query_context()`:
```python
def _build_query_context(self, question, retrieved) -> tuple[str, list[dict]]:
    # 统一的 context string + sources list 构建逻辑
```

**测试方法**: `POST /api/query` 和 `POST /api/query/stream` 返回的 sources 结构一致。

---

### 4. 异步 Reranker（不阻塞事件循环）

**问题**: `CrossEncoder.predict()` 是同步 CPU 密集型调用，在 async 上下文中阻塞整个事件循环。

**文件**: `app/reranker.py` + `app/agent/qa_agent.py`

**改动**: `arerank()` 用 `asyncio.to_thread()` 将重排放到线程池:
```python
async def arerank(self, query, documents, top_k=None):
    return await asyncio.to_thread(self._rerank_sync, query, documents, top_k)
```

**测试方法**: 同时发 5 个 query 请求，检查响应时间均匀（而非串行等待）。

---

### 5. Agent 工具调用集成

**问题**: `tools.py` 定义了 web_search/calculator/current_time 但从未被 QA 流程使用。

**文件**: `app/agent/qa_agent.py`

**改动**:
- `query()` 中改用 `_llm_invoke_with_tools()` 替代直接 `achat()`
- `_llm_invoke_with_tools()`: LLM 响应中检测 `{"tool": "...", "arguments": {...}}` JSON，自动调用工具并二次请求 LLM（最多 2 轮）
- `_parse_tool_call()`: 正则提取 LLM 输出中的工具调用 JSON
- `SYSTEM_PROMPT` 和 `TOOL_AWARE_SYSTEM_PROMPT` 更新工具说明

**测试方法**:
```
POST /api/query  {"question": "现在几点了？"}
— 应返回当前时间（即使知识库中没有相关内容）

POST /api/query  {"question": "计算 123*456"}
— 应返回计算结果 56088
```

---

### 6. 查询结果缓存

**问题**: `config.py` 定义了 `cache_enabled/cache_ttl/cache_max` 但无实际实现。

**文件**: `app/agent/qa_agent.py`

**改动**:
- `__init__()`: 内存 dict 作为 LRU 缓存
- `_cache_get()`: 检查 TTL，过期自动淘汰
- `_cache_set()`: 写入缓存，超出 max 时淘汰最旧条目
- `query()`: 检索前查缓存，LLM 调用后写缓存
- `get_stats()`: 返回 `cache_size/hits/misses/max`

**测试方法**:
1. 发同一个问题两次，第二次响应应极快（缓存命中）
2. `GET /api/stats` 中 `cache_hits` > 0

---

### 7. 去除 LangChain 依赖

**问题**: `langchain` 4 个包约 80MB，大部分功能未使用。

**文件**:
- `app/agent/llm_client.py`: `ChatOpenAI` -> `AsyncOpenAI` 直连
- `app/agent/qa_agent.py`: `RecursiveCharacterTextSplitter` -> 自研 `_SimpleTextSplitter`
- `requirements.txt`: 移除 4 个 langchain 包

**测试方法**:
```
pip install -r requirements.txt  # 不应安装 langchain 相关包
python run.py                    # 启动正常
POST /api/documents/text         # 文档分块正常
POST /api/query                  # LLM 调用正常
```

---

### 8. PyMuPDF 替换 pdfminer.six

**文件**: `requirements.txt`

**改动**: `pdfminer.six>=20221105` -> `PyMuPDF>=1.24.0`（`utils.py` 中已用 fitz，现在补上依赖声明）

**测试方法**: 上传 PDF 文件 `POST /api/documents/upload`，返回 tables 和 images 数量。

---

### 9. 前端缓存统计显示

**文件**: `app/templates/index.html`

**改动**:
- 侧边栏底部新增 `#cacheStats` 显示缓存命中率
- `loadStats()` 中更新缓存状态
- 页面加载时自动调用 `loadStats()`

**测试方法**: 打开 `http://localhost:8020`，左侧栏底部应显示缓存统计。

---

## 第二轮: 安全与可靠性

### 10. Windows/Linux 跨平台兼容

**问题**: `run.py` 硬编码 `loop="uvloop"` 和 `http="httptools"`，Windows 上直接崩溃。

**文件**: `run.py`

**改动**: 平台检测函数 `_detect_loop()`:
```python
def _detect_loop():
    if platform.system() == "Windows":
        return "asyncio"
    try:
        import uvloop
        return "uvloop"
    except ImportError:
        return "asyncio"
```

**测试方法**: Windows 上 `python run.py` 应正常启动。

---

### 11. 速率限制异步化

**问题**: `rate_limit.py` 使用 `threading.Lock()`，在 async 中间件中阻塞整个事件循环。

**文件**: `app/middleware/rate_limit.py`

**改动**:
- `TokenBucket._lock`: `threading.Lock` -> `asyncio.Lock`
- `TokenBucket.allow()`: 改为 `async def`
- `RateLimitMiddleware._get_bucket()`: 改为 async
- `RateLimitMiddleware._cleanup()`: 改为 async
- `dispatch`: `bucket.allow()` -> `await bucket.allow()`
- 新增 `/metrics` 路径豁免

**测试方法**: 并发压测 `/api/query`，第 61 个请求应返回 429。

---

### 12. Docker 非 root 运行

**问题**: `USER nobody` 无法写 `/app/data/`、`/app/logs/`、`/app/knowledge/` 目录。

**文件**: `Dockerfile`

**改动**:
- 创建专用用户 `appuser:appuser`
- `chown -R appuser:appuser /app` 确保所有目录可写
- `HEALTHCHECK` 启动等待从 90s -> 120s（模型加载需要时间）

**测试方法**:
```bash
docker build -t securtyagent .
docker run -p 8020:8020 securtyagent
curl http://localhost:8020/api/health
# 应返回 OK
```

---

### 13. Kubernetes 健康检查

**问题**: 只有 `/api/health`，无法区分 readiness 和 liveness。

**文件**: `app/routers/api.py`

**改动**:
- `/api/health`: 分别检查 SQLite、ChromaDB、LLM 三个组件，返回 `ok/degraded/unhealthy`
- `/api/health/ready`: readiness probe — 所有后端必须可用
- `/api/health/live`: liveness probe — 进程存活

**测试方法**:
```bash
curl http://localhost:8020/api/health/ready  # -> {"status":"ready"}
curl http://localhost:8020/api/health/live   # -> {"status":"alive"}
```

---

### 14. 模型加载超时保护 + 错误处理

**问题**: 模型加载无异常处理，下载失败时无限挂起。

**文件**: `app/embeddings.py` + `app/config.py` + `.env.example`

**改动**:
- `_load_model()` 添加 try/except，失败时抛出 `EmbeddingError`
- 记录加载开始/结束时间
- 支持 `huggingface_token` 环境变量（用于 gated 模型）
- `config.py` 新增 `huggingface_token` 字段
- `.env.example` 新增 `HF_TOKEN` 示例

**测试方法**: 故意填错 `EMBEDDING_MODEL` 为不存在模型名，启动应报清晰的错误日志。

---

### 15. 日志脱敏 — API Key 不泄漏

**问题**: LLM SDK 报错信息可能包含 API Key。

**文件**:
- `app/agent/llm_client.py`: `achat()` 最终异常处理中对 key 脱敏（替换为 `***`）
- `app/main.py`: 全局异常 handler 对 `settings.llm_api_key` 做 `[REDACTED]` 替换

**测试方法**: 故意用无效 API Key 发请求，检查日志中不含实际 key 值。

---

### 16. CSP 安全头增强

**文件**: `app/middleware/security.py`

**改动**:
- 新增 `Content-Security-Policy` 头（default-src + script-src + style-src + img-src + connect-src）
- 隐藏 `Server` 响应头

**测试方法**: `curl -I http://localhost:8020` 应看到 CSP 头，无 Server 头。

---

### 17. 资源优雅释放

**文件**: 
- `app/database/vector_store.py` — `VectorStore.close()` 增加 `del self._chroma_client`
- `app/main.py` — `lifespan()` shutdown 阶段增加清理框架

---

## 第三轮: 细节打磨

### 18. SVD 懒重建（避免每次 add 全量重算）

**问题**: `semantic_store.py` 中 `SemanticIndex` 每次 `add_texts()` 都触发全量 SVD 重算 O(n3)。文档 200+ 时一次 add 卡 5-15 秒。

**文件**: `app/database/semantic_store.py`

**改动**:
- `add_texts()`: 只写 `docs.json`，设置 `_fitted = False`，不再立即重建
- `_ensure_fitted()`: 在 `search()` 时懒重建，只在必要时算一次
- `delete_by_doc_id()`: 删除后立即重建（需要反映删除效果）
- `_save()` 拆分为 `_save_docs()`（轻量）和 `_save()`（完整，rebuild 后调用）
- 新增 `_pending_adds` 和 `_addition_count` 状态追踪

**测试方法**: 连续上传 5 个文档，总时间应明显低于优化前。

---

### 19. MarkdownStore 文件名防覆盖

**问题**: `_safe_filename()` 同名标题直接覆盖旧文件。

**文件**: `app/database/markdown_store.py`

**改动**: `save()` 方法中检测 `filepath.exists()`，若存在则加 `_YYYYMMDDHHMMSS` 后缀。

**测试方法**: 两次添加相同标题的文档，检查 `knowledge/` 目录中生成两个文件而非一个。

---

### 20. API 分页

**问题**: `GET /api/documents` 无分页参数，1000 个文档时返回几 MB JSON。

**文件**: `app/routers/api.py`

**改动**:
```python
@router.get("/documents")
async def list_documents(
    category: str = Query(default=""),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
):
```

**测试方法**: `GET /api/documents?offset=0&limit=5` 应只返回 5 条。

---

### 21. delete_document 顺序修正

**问题**: 先删 SQLite（核心数据）-> 再删 vector（可能失败），导致数据不一致。

**文件**: `app/agent/qa_agent.py`

**改动**: 先删 vector store（best-effort）-> 原子删 SQLite（核心保证）-> 删 markdown（best-effort）

---

### 22. Query Rewrite 智能跳过

**问题**: 每次 query 都调 LLM 改写，即使只有一条新问题、无历史、无代词。浪费 token 和时间。

**文件**: `app/agent/query_processor.py`

**改动**:
- 无历史（`history < 2`）直接返回原问题
- 有历史但问题不含代词指标词（它/他/她/这个/那个/这些/那些/刚才/继续/再）跳过 LLM
- `rewritten` 结果为空时回退到原问题

**测试方法**: 第一条问题 `POST /api/query {"question":"什么是AI"}` 不应触发 query rewrite 日志。

---

### 23. BM25 加载性能优化

**问题**: `_load()` 中逐文档遍历 tokens 重建倒排索引，O(n) token 遍历。

**文件**: `app/database/vector_store.py` — `BM25Index._load()`

**改动**: 直接从 `posting` 表读取并批量重建倒排索引:
```python
post_rows = conn.execute(
    "SELECT token, doc_idx, position FROM posting ORDER BY token, doc_idx"
).fetchall()
```

**测试方法**: 启动日志中 `BM25 index loaded from SQLite: N docs, M tokens` 出现且加载时间 < 1s。

---

### 24. StatsResponse Schema 补全

**文件**: `app/models/schemas.py`

**改动**: `StatsResponse` 新增 `cache_size/cache_hits/cache_misses/cache_max` 字段（均带默认值 0）

**测试方法**: `GET /api/stats` 返回的 JSON 包含 cache 相关字段。

---

### 25. XSS 防御增强

**问题**: `esc()` 函数未转义引号。

**文件**: `app/templates/index.html`

**改动**: `esc()` 增加 `"` 和 `'` 转义

---

## 修改文件清单

| 文件 | 轮次 | 改动类型 |
|------|------|---------|
| `app/main.py` | 1,2 | 预热、日志脱敏、关闭框架 |
| `app/agent/qa_agent.py` | 1,3 | 去重、缓存、工具调用、text splitter、原子删除 |
| `app/agent/llm_client.py` | 1,2 | AsyncOpenAI 直连、日志脱敏 |
| `app/agent/query_processor.py` | 3 | 智能跳过 LLM 改写 |
| `app/database/vector_store.py` | 1,2,3 | BM25 SQLite 增量、资源释放、加载优化 |
| `app/database/semantic_store.py` | 3 | SVD 懒重建 |
| `app/database/markdown_store.py` | 3 | 文件名防覆盖 |
| `app/reranker.py` | 1 | 异步 arerank() |
| `app/embeddings.py` | 2 | 超时保护、HF token |
| `app/config.py` | 2 | huggingface_token 字段 |
| `app/routers/api.py` | 2,3 | 健康检查、分页 |
| `app/models/schemas.py` | 3 | StatsResponse 补全 |
| `app/middleware/rate_limit.py` | 2 | asyncio.Lock 异步化 |
| `app/middleware/security.py` | 2 | CSP 安全头、隐藏 Server |
| `app/templates/index.html` | 1,3 | 缓存统计显示、XSS 增强 |
| `run.py` | 2 | Windows/Linux 平台检测 |
| `Dockerfile` | 2 | appuser 用户与权限 |
| `requirements.txt` | 1 | 去 langchain 群组、加 PyMuPDF |
| `.env.example` | 2 | HF_TOKEN 示例 |

---

## 快速测试命令

```bash
# 启动
python run.py

# 健康检查
curl http://localhost:8020/api/health
curl http://localhost:8020/api/health/ready
curl http://localhost:8020/api/health/live

# 上传文档
curl -X POST http://localhost:8020/api/documents/text \
  -F "title=测试" -F "content=这是一个测试文档" -F "tags=test"

# 查询（首次）
curl -X POST http://localhost:8020/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"测试文档"}'

# 查询（第二次应命中缓存）
curl -X POST http://localhost:8020/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"测试文档"}'

# 工具调用
curl -X POST http://localhost:8020/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"现在几点了"}'

# 统计
curl http://localhost:8020/api/stats

# 分页
curl "http://localhost:8020/api/documents?offset=0&limit=5"

# 前端
# 浏览器打开 http://localhost:8020
```
