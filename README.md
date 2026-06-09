# Knowledge QA Agent

企业级 RAG（检索增强生成）知识问答系统，支持多格式文档导入、混合检索、LLM 生成回答的全链路闭环。

[![Python](https://img.shields.io/badge/python-3.14-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 架构

```
用户提问 → Query Rewrite → 混合检索 → Rerank → LLM 生成 → 回答
                              ↓
              ┌───────────────┼───────────────┐
         ChromaDB          BM25          Semantic Index
        (语义向量)       (关键词)         (TF-IDF + SVD)
                              ↓
                          SQLite (元数据)
                          Markdown (原始文档)
```

## 核心特性

### 检索
- **三重索引**：ChromaDB 向量检索 + BM25 关键词索引 + TF-IDF 语义索引（SVD 降维）
- **交叉编码器重排**：BGE Reranker v2-m3 对召回结果精排
- **Query Rewrite**：智能判断是否需要 LLM 改写，节省 Token

### 文档管理
- 支持文本录入和文件上传（PDF/TXT/Markdown）
- 自动分块、向量化、语义索引
- Markdown 原始文件存储，可直接阅读

### 生产化
- 异步架构：FastAPI + asyncio，Reranker 在线程池运行不阻塞事件循环
- **缓存加速**：LRU 查询缓存，重复查询 10.5s → 29ms（361 倍提升）
- 速率限制（asyncio.Lock）、K8s 健康检查、CSP 安全头、日志脱敏
- Windows/Linux 跨平台

### LLM 集成
- 直连 OpenAI 协议（DeepSeek/Qwen 等），无 LangChain 依赖
- 工具调用（计算器/当前时间）
- 流式 Streaming 响应

## 快速开始

```bash
pip install -r requirements.txt

# 设置环境变量
set DEEPSEEK_API_KEY=your_api_key

# 启动服务
python run.py

# 打开浏览器 http://localhost:8020
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/query` | 提问（支持 streaming） |
| POST | `/api/query/stream` | 流式提问 |
| POST | `/api/documents/text` | 添加文本文档 |
| POST | `/api/documents/upload` | 上传文件 |
| GET | `/api/documents` | 文档列表（支持分页） |
| DELETE | `/api/documents/{id}` | 删除文档 |
| GET | `/api/stats` | 系统统计 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/health/ready` | Readiness Probe |
| GET | `/api/health/live` | Liveness Probe |

## 性能测试

| 场景 | 指标 |
|---|---|
| 首次查询（含检索+LLM） | ~10.5s |
| 缓存命中查询 | **29ms**（361x 提升） |
| 跨领域检索准确率 | >0.75（语义匹配 score） |
| 无相关文档拒答 | score ≈ 0.004 |
| 并发 3 查询 | 总耗时 17.9s |
| 模型冷启动（嵌入+重排） | ~25s（预热后） |

## 项目结构

```
├── app/
│   ├── agent/          # QA Agent 核心逻辑
│   │   ├── qa_agent.py     # 主流程：检索 → LLM → 回答
│   │   ├── llm_client.py   # LLM 直连客户端
│   │   ├── query_processor.py  # Query Rewrite
│   │   ├── conversation.py # 对话管理
│   │   └── tools.py        # 工具调用
│   ├── database/       # 三种存储实现
│   │   ├── vector_store.py  # ChromaDB + BM25
│   │   ├── semantic_store.py  # SVD 语义索引
│   │   ├── sql_store.py    # SQLite 元数据
│   │   └── markdown_store.py  # 文件存储
│   ├── middleware/     # 中间件（安全、限流、认证）
│   ├── routers/        # API 路由
│   └── templates/      # Web UI
├── tests/              # 测试
└── knowledge/          # Markdown 文档存储
```
