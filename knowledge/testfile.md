# testfile

> Tags: test

> Created: 2026-06-10T09:29:05.422395+00:00

---

# Knowledge QA Agent

企业级 RAG（检索增强生成）知识问答系统 — 三重混合检索 + Cross-Encoder 重排 + LLM 生成，生产化就绪。

[![Python](https://img.shields.io/badge/python-3.14-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-purple)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 技术栈

| 类别 | 技术 |
|---|---|
| **Web 框架** | FastAPI + Uvicorn + Pydantic |
| **向量数据库** | ChromaDB（语义检索） |
| **关键词索引** | BM25（SQLite 增量存储） |
| **语义索引** | TF-IDF + SVD 降维（LSI） |
| **Embedding** | BAAI/bge-small-zh-v1.5（本地） |
| **Reranker** | BAAI/bge-reranker-v2-m3（Cross-Encoder） |
| **LLM** | DeepSeek / OpenAI 兼容 API（直连，无 LangChain） |
| **存储** | SQLite（元数据 + 分块）、Markdown（原始文档） |
| **生产化** | CSP 安全头、速率限制、日志脱敏、K8s 健康检查、Docker |

## 架构

```
                        用户提问
                           │
                           ▼
                    ┌──────────────┐
                    │ Query Rewrite│  智能跳过（无代词/短历史时直接透传）
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   混合检索     │
                    │               │
            ┌───────┼───────┬───────┼───────┐
            │       │       │       │       │
        ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐    │
        │向量  │ │BM25 │ │标签  │ │语义  │    │
        │检索  │ │关键词│ │查询  │ │索引  │    │
        │Chroma│ │SQLite│ │SQLite│ │SVD  │    │
        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘    │
            │       │       │       │       │
            └───────┴───────┴───────┴───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Cross-Encoder│  BGE Reranker v2-m3
                    │   重排 Top-K  │  (asyncio.to_thread 异步)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LLM 生成回答 │  DeepSeek 直连
                    │  + 工具调用  │  计算器 / 时间 / 搜索
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LRU 查询缓存 │  过期淘汰，361x 加速
                    └──────────────┘
```

## 核心特性

### 三重混合检索

| 检索方式 | 技术 | 特点 |
|---|---|---|
| **语义向量** | ChromaDB + BGE Embedding | 理解语义相似度，处理同义词 |
| **关键词** | BM25（SQLite 增量存储） | 精确匹配，对专有名词友好 |
| **语义索引** | TF-IDF + SVD（LSI） | 降维去噪，兼顾语义与词频 |

检索结果经 **Cross-Encoder 重排**（BGE Reranker v2-m3）精排后输入 LLM。

### 文档管理
- **文本录入**：直接输入内容，自动分块
- **文件上传**：支持 PDF / TXT / Markdown，自动提取内容
- **分块存储**：向量库（ChromaDB）+ 关键词库（BM25 SQLite）+ 语义索引（SVD）+ 元数据库（SQLite）+ 原始文件（Markdown）

### 生产化改造
- **BM25 增量存储**：SQLite 逐条 INSERT 替代全量 JSON 序列化
- **SVD 懒重建**：仅在查询时触发，避免每次添加文档全量重算 O(n³)
- **异步 Reranker**：`asyncio.to_thread` 将 CPU 密集型重排放到线程池
- **模型预热**：启动时预加载 Embedding + Reranker，避免首请求阻塞
- **速率限制**：`asyncio.Lock` 令牌桶，支持并发安全
- **缓存**：LRU 内存缓存，TTL 过期自动淘汰
- **日志脱敏**：API Key 在日志中自动替换为 `***`

### 部署
- **Docker** 多阶段构建，非 root 用户运行
- **K8s 健康检查**：liveness（进程存活）/ readiness（组件可用）/ health（详细状态）
- **安全头**：CSP（Content-Security-Policy）、隐藏 Server 头
- **跨平台**：Windows / Linux 自适应事件循环

### LLM 集成
- 直连 OpenAI Chat Completions API（DeepSeek / Qwen 等），无 LangChain 依赖
- 工具调用（JSON 格式提取：计算器、当前时间、搜索）
- 流式 Streaming 响应（Server-Sent Events）
- 最大 2 轮工具调用循环

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
set DEEPSEEK_API_KEY=your_api_key

# 3. 启动服务
python run.py

# 4. 浏览器打开 http://localhost:8020
```

国内网络需设置 HuggingFace 镜像：
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

## API 参考

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/query` | 提问（返回 JSON） |
| POST | `/api/query/stream` | 流式提问（SSE） |
| POST | `/api/documents/text` | 添加文本文档 |
| POST | `/api/documents/upload` | 上传文件（PDF/TXT/MD） |
| GET | `/api/documents` | 文档列表（分页参数：offset, limit） |
| DELETE | `/api/documents/{id}` | 删除文档 |
| GET | `/api/stats` | 系统统计（文档数、缓存命中率） |
| GET | `/api/health` | 详细健康检查 |
| GET | `/api/health/ready` | K8s Readiness Probe |
| GET | `/api/health/live` | K8s Liveness Probe |

## 性能测试

| 场景 | 指标 |
|---|---|
| 首次查询（含检索 + LLM 生成） | ≈ 10.5s |
| 缓存命中查询 | **29ms**（361× 提升） |
| 跨领域检索准确率 | > 0.75（语义匹配 score） |
| 无相关文档拒答能力 | score ≈ 0.004（正确低分） |
| 并发 3 查询总耗时 | 17.9s |
| 模型冷启动（Embedding + Reranker） | ≈ 25s（预热后） |
| 缓存容量 | LRU max=256，TTL 自动淘汰 |

## 项目结构

```
├── app/
│   ├── agent/                  # QA Agent 核心
│   │   ├── qa_agent.py         # 主流程：检索 → 重排 → LLM → 回答
│   │   ├── llm_client.py       # LLM 直连客户端（AsyncOpenAI）
│   │   ├── query_processor.py  # Query Rewrite 智能改写
│   │   ├── conversation.py     # 对话上下文管理
│   │   └── tools.py            # 工具调用（计算器 / 时间）
│   ├── database/               # 存储层
│   │   ├── vector_store.py     # ChromaDB + BM25（SQLite 增量）
│   │   ├── semantic_store.py   # TF-IDF + SVD 语义索引
│   │   ├── sql_store.py        # SQLite 结构化存储
│   │   └── markdown_store.py   # 原始 Markdown 文件存储
│   ├── middleware/             # 中间件
│   │   ├── auth.py             # API Key 认证
│   │   ├── rate_limit.py       # 令牌桶速率限制（asyncio）
│   │   └── security.py         # CSP 安全头、隐藏 Server
│   ├── models/
│   │   └── schemas.py          # Pydantic 数据模型
│   ├── routers/
│   │   └── api.py              # API 路由定义
│   ├── templates/
│   │   └── index.html          # Web 管理界面
│   ├── config.py               # Pydantic Settings 配置
│   ├── embeddings.py           # BGE Embedding 加载与预热
│   ├── reranker.py             # BGE Reranker 异步封装
│   ├── monitoring.py           # 日志与监控
│   ├── exceptions.py           # 异常定义
│   └── utils.py                # 工具函数
├── tests/                      # 测试
│   ├── test_agent.py
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_integration.py
│   └── test_markdown_store.py
├── knowledge/                  # Markdown 文档存储目录
├── run.py                      # 生产入口（uvicorn）
├── run_server.py               # 开发入口
├── Dockerfile                  # Docker 多阶段构建
├── docker-compose.yml          # Docker Compose
├── pyproject.toml              # 项目元数据
└── requirements.txt            # Python 依赖
```

## 依赖

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
chromadb>=0.6.0
sentence-transformers>=3.0.0
pillow>=11.0.0
PyMuPDF>=1.24.0
httpx>=0.28.0
aiofiles>=24.0.0
```
