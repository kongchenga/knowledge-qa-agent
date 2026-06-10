# P0 检索精度升级完成 - 改进说明

## 已完成改动

### 1. 新增语义检索引擎 pp/database/semantic_store.py
- 基于 TF-IDF + TruncatedSVD (LSI) 的轻量语义向量检索
- 无需外部 AI 模型，纯 sklearn + numpy 实现
- 中英文混合文本支持，自定义 tokenizer

### 2. 升级为三路混合检索 pp/agent/qa_agent.py
- **BM25**（稀疏检索，关键词精确匹配）
- **LSI 语义**（稠密检索，语义相似度）
- **SQLite 标签**（结构化标签查询）
- **RRF 融合**（Reciprocal Rank Fusion 合并三路结果）
- **LLM 重排**（DeepSeek 智能排序）

### 3. 配置更新 pp/config.py
- LSI_ENABLED / LSI_N_COMPONENTS — 语义检索开关/维度
- LLM_RERANK_ENABLED / LLM_RERANK_TOP_K / LLM_RERANK_FINAL_K — LLM 重排配置

### 4. Bug 修复
- BM25 索引的持久化加载（原来缺少 _load 方法）
- IDF 计算中的文档频率统计错误（df 应为唯一 doc 数而非 posting 总数）

### 5. 前端显示增强 pp/templates/index.html
- 统计面板显示 BM25/LSI 分块数和检索状态
- 来源显示检索方式标识

## 本地依赖
pylib/ 目录下安装了 onnxruntime、scipy、scikit-learn、joblib 等，已加入 sys.path。

## 后续待办 (P1-P4)
- **P1**: PDF/Office 深度文档解析（表格、图片保留）
- **P2**: 多轮对话 + 知识图谱
- **P3**: Agent 工具调用能力
- **P4**: 前端管理界面完善

---

## 下一步：P1-P4 改进计划

详见 TODO_P1_P2.md（P1-P2）和下方说明（P3-P4）。

### P3: Agent 工具调用
- 接入 DeepSeek function calling
- 工具：Web 搜索、计算器、当前时间等
- 多步推理（ReAct 模式）

### P4: 前端管理界面
- 文档管理页面（上传、编辑、删除）
- 对话历史展示
- 知识库分类和权限管理
