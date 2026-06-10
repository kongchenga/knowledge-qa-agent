# P1: 文档解析升级

## 目标
提升 PDF、Office 文件的解析质量，保留表格、图片、格式信息。

## 具体任务

### 1.1 PDF 解析增强
- 使用 `PyMuPDF` (fitz) 替代 pdfminer.six
- 保留表格结构（提取为 Markdown 表格）
- 提取图片并保存到 `knowledge/images/` 目录
- 提取标题层级（h1/h2/h3）

### 1.2 Office 文件解析
- `.docx`: 用 `python-docx` 保留格式、表格、图片
- `.xlsx`: 用 `openpyxl` 提取表格数据

### 1.3 修改文件
- `app/utils.py` — 新增 `extract_pdf_with_layout()`, `extract_docx_with_format()`
- `app/routers/api.py` — 上传接口返回图片/表格数量
- `app/templates/index.html` — 显示文档解析详情

---

# P2: 多轮对话

## 目标
支持多轮对话上下文，让 LLM 能记住前文。

## 具体任务

### 2.1 对话历史管理
- 在 SQLite 中新建 `conversations` 表和 `messages` 表
- 每次 query 时传入 `session_id` 参数
- 自动携带最近 N 轮对话作为上下文

### 2.2 修改文件
- `app/database/sql_store.py` — 添加对话历史表
- `app/models/schemas.py` — 新增 QueryRequest 增加 session_id 字段
- `app/agent/qa_agent.py` — query 方法增加历史上下文拼接
- `app/templates/index.html` — 前端多轮对话展示
