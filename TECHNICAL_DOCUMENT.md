# Stock Agent - 项目技术文档

## 1. 项目概述

**项目名称**：Stock Agent - 智能股票分析系统  
**项目类型**：毕业设计 + 求职作品  
**开发周期**：2025年X月 - 2026年X月  
**项目简介**：基于大语言模型（LLM）的 A 股智能分析平台，集成 AI Agent 对话、知识库检索（RAG）、K 线图表生成、用户管理等功能，提供智能化的股票分析和投资建议。

---

## 2. 技术架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit 前端层                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ 登录/注册 │  │ AI 助手  │  │ 数据仪表盘│  │ 知识库   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent 编排层 (LangChain)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  BaseAgent (ReAct 模式)                               │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────┐   │   │
│  │  │ 系统提示词  │ │ 工具列表   │ │ 短期记忆       │   │   │
│  │  └────────────┘ └────────────┘ └────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  股票数据工具   │ │  RAG 检索工具  │ │  时间工具     │
│  (Tushare)    │ │  (Chroma)     │ │              │
└───────────────┘ └───────────────┘ └───────────────┘
            │               │
            ▼               ▼
┌───────────────┐ ┌───────────────┐
│  PostgreSQL   │ │  ChromaDB     │
│  (业务数据)    │ │  (向量数据)    │
└───────────────┘ └───────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM 服务层 (Ollama)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  LLM 模型    │  │ Embedding 模型│  │  (可扩展)     │      │
│  │  qwen3.6     │  │ nomic-embed  │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈详情

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **前端** | Streamlit | >=1.45.0 | Web UI 框架，快速构建数据应用 |
| **AI 框架** | LangChain | >=0.3.25 | Agent 编排、工具调用、记忆管理 |
| **LLM** | Ollama (qwen3.6) | - | 本地部署大语言模型 |
| **Embedding** | Ollama (nomic-embed-text) | - | 文本向量化 |
| **向量数据库** | ChromaDB | >=1.0.0 | 向量存储与相似度检索 |
| **数据源** | Tushare Pro | >=0.12.0 | A 股历史行情数据 |
| **关系数据库** | PostgreSQL | 16+ | 用户、会话、消息存储 |
| **ORM** | SQLAlchemy | >=2.0.0 | 数据库对象关系映射 |
| **密码加密** | bcrypt | >=4.0.0 | 用户密码哈希存储 |
| **PDF 解析** | pypdf | >=5.0.0 | 知识库文档加载 |
| **图表** | Matplotlib | >=3.10.0 | K 线图、趋势图生成 |
| **日志** | Loguru | >=0.7.0 | 高性能日志框架 |
| **配置** | Pydantic Settings | >=2.0.0 | 环境变量与配置管理 |
| **部署** | Uvicorn | >=0.34.0 | ASGI 服务器 |

---

## 3. 核心功能模块

### 3.1 AI Agent 系统

#### 3.1.1 Agent 架构设计

采用 **ReAct（Reasoning + Acting）** 模式的 Agent 架构，通过 LangChain 的 `create_agent` 实现：

```python
# 核心创建逻辑
def create_agent(preset: str = "default") -> BaseAgent:
    llm = get_llm(preset)
    agent_executor = create_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=MemorySaver(),  # 短期记忆
    )
    return BaseAgent(agent_executor, config)
```

**设计亮点**：
- **预设配置系统**：支持 4 种预设模式（default/precise/creative/structure），运行时切换不同参数组合
- **短期记忆**：基于 LangChain `MemorySaver` 实现会话级记忆，支持多轮对话
- **流式输出**：支持 `stream()` 模式，实时返回推理过程和工具调用

#### 3.1.2 工具系统

| 工具 | 功能 | 输入参数 |
|------|------|----------|
| `get_stock_daily_data` | 获取股票历史行情 | ts_code, days |
| `get_stock_basic_info` | 获取股票基本信息 | ts_code |
| `plot_stock_charts` | 绘制 K 线/趋势/饼图 | ts_code, stock_name |
| `search_knowledge_base` | 检索投资知识库 | query, top_k |
| `get_current_datetime` | 获取当前时间 | 无 |

**技术指标计算**：
- MA5/20/60（移动平均线）
- 日收益率与波动率
- 成交量统计与分析

### 3.2 RAG 知识库系统

#### 3.2.1 RAG 架构

```
用户上传文档 (PDF/TXT)
        │
        ▼
┌──────────────┐
│ 文档加载器    │  pypdf / TextLoader
└──────────────┘
        │
        ▼
┌──────────────┐
│ 文本分块器    │  RecursiveCharacterTextSplitter
│ chunk_size=500│  chunk_overlap=50
└──────────────┘
        │
        ▼
┌──────────────┐
│ Embedding    │  Ollama nomic-embed-text (本地)
└──────────────┘
        │
        ▼
┌──────────────┐
│ ChromaDB     │  向量持久化存储
└──────────────┘
```

#### 3.2.2 RAG 检索流程

采用 **RAG as Tool** 模式，Agent 自主决定何时调用知识库检索：

```
用户提问："什么是 MACD 指标？"
        │
        ▼
Agent 推理 → 需要调用 search_knowledge_base
        │
        ▼
ChromaDB 向量检索 → 返回相关文档片段
        │
        ▼
Agent 综合检索结果 + 自身知识 → 生成回答
```

#### 3.2.3 知识库管理

- **预置知识库**：系统内置投资分析、技术指标、风险管理等专业知识
- **用户自定义**：支持用户上传 PDF/TXT 文档扩展知识库
- **集合隔离**：预置和用户知识库存储在不同 ChromaDB collection 中

### 3.3 股市数据仪表盘

#### 3.3.1 功能概述

独立的数据可视化页面，无需 AI 对话即可快速浏览市场行情：

| 模块 | 内容 | 数据来源 |
|------|------|----------|
| **主要指数** | 上证指数、深证成指、创业板指、恒生指数、标普500 | Tushare / 模拟 |
| **K 线图** | 上证指数近 60 交易日 K 线 + MA5/MA20 + 成交量 | Tushare / 模拟 |
| **市场概况** | 上涨/下跌/平盘家数 + 总成交额 | Tushare / 模拟 |
| **涨跌饼图** | 上涨/下跌/平盘占比可视化 | 基于市场概况 |

#### 3.3.2 数据获取策略

```python
def _fetch_index_daily(ts_code: str) -> dict | None:
    """A 股指数：Tushare index_daily API"""
    df = pro.index_daily(ts_code=ts_code, ...)
    return {"close": ..., "change_pct": ...}

def _generate_mock_index(name: str) -> dict:
    """港股/美股：模拟数据（标注 [模拟] 标签）"""
    return {"close": ..., "change_pct": ..., "is_mock": True}
```

- A 股指数（上证/深证/创业板）：通过 Tushare `index_daily` 获取真实数据
- 境外指数（恒生/标普500）：生成模拟数据，界面标注 `[模拟]`
- 市场概况：通过 Tushare `daily` 统计涨跌家数，回退到模拟数据

#### 3.3.3 图表实现

- **K 线图**：Matplotlib 手动绘制蜡烛图，红涨绿跌，暗色主题适配
- **涨跌饼图**：Matplotlib `pie()` 函数，分离式显示（explode）

### 3.4 用户与会话管理系统

#### 3.4.1 数据模型

```python
class User(Base):
    id: Integer (PK)
    username: String (unique)
    password_hash: String (bcrypt)
    is_admin: Boolean
    created_at: DateTime

class Conversation(Base):
    id: Integer (PK)
    user_id: Integer (FK → users)
    title: String
    agent_type: String ('analysis' / 'chat')
    created_at: DateTime
    updated_at: DateTime

class Message(Base):
    id: Integer (PK)
    conversation_id: Integer (FK → conversations)
    role: String ('human' / 'ai')
    content: Text
    chart_paths: JSON (图表路径列表)
    created_at: DateTime
```

#### 3.4.2 会话持久化流程

```
用户发送消息 → Agent 处理 → 保存 Message 到 DB
                    ↓
              流式输出响应
                    ↓
         保存 AI 回复 + 图表路径到 DB
                    ↓
         更新 Conversation 的 updated_at
```

### 3.5 股票数据可视化（AI 助手）

#### 3.5.1 图表类型

AI 助手对话中自动生成的图表：

1. **K 线图**：包含 MA5/20/60 均线叠加，标注涨跌
2. **趋势图**：收盘价 + MA5/20/60 趋势线 + 成交量
3. **成交量分布图**：饼图展示成交量区间分布

#### 3.5.2 图表管理

- 图表存储在 `imgs/stock/` 目录
- 文件名格式：`{code}_{timestamp}.png`
- 会话中图表路径随消息持久化存储

---

## 4. 关键技术亮点

### 4.1 多 LLM 实例管理

**场景**：本地有 embedding 模型但性能较弱，服务器有强大的 LLM 但无 embedding。

**解决方案**：
- LLM 调用走服务器（192.168.8.21:11434）
- Embedding 调用走本地（127.0.0.1:11434）
- 通过配置分离管理：

```python
# utils/setting.py
llm_base_url: str = "http://192.168.8.21:11434"      # 服务器 LLM
embedding_base_url: str = "http://127.0.0.1:11434"   # 本地 Embedding
```

### 4.2 Agent 预设配置系统

支持 4 种预设模式，运行时切换：

| 预设 | Temperature | Max Tokens | 适用场景 |
|------|-------------|------------|----------|
| default | 0.1 | 4096 | 日常对话 |
| precise | 0 | 2048 | 精确分析 |
| creative | 0.8 | 8192 | 创意生成 |
| structure | 0.1 | 4096 | 结构化输出 |

### 4.3 流式输出与记忆管理

```python
# 支持流式输出
for chunk in agent.stream(input, config):
    yield chunk

# 短期记忆基于 LangGraph checkpointer
checkpointer = MemorySaver()
agent = create_agent(..., checkpointer=checkpointer)
```

### 4.4 RAG as Tool 模式

Agent 自主决定何时调用知识库，而非每次查询都检索：

```python
@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """当用户询问投资知识、技术指标或需要专业知识时调用"""
    collection = get_collection("preset")
    results = collection.query(query_texts=[query], n_results=top_k)
    return "\n\n".join(results["documents"][0])
```

---

## 5. 数据库设计

### 5.1 ER 图

```
┌─────────────┐       ┌─────────────────┐       ┌─────────────┐
│    User     │ 1   n │  Conversation   │ 1   n │   Message   │
├─────────────┤───────├─────────────────┤───────├─────────────┤
│ id (PK)     │       │ id (PK)         │       │ id (PK)     │
│ username    │       │ user_id (FK)    │       │ conversation│
│ password_   │       │ title           │       │ _id (FK)    │
│   hash      │       │ agent_type      │       │ role        │
│ is_admin    │       │ created_at      │       │ content     │
│ created_at  │       │ updated_at      │       │ chart_paths │
└─────────────┘       └─────────────────┘       │ created_at  │
                                                └─────────────┘
```

### 5.2 索引设计

- `users.username`：唯一索引，加速登录查询
- `conversations.user_id`：普通索引，加速会话列表查询
- `messages.conversation_id`：普通索引，加速消息加载

---

## 6. 配置管理

### 6.1 环境变量

```env
# LLM 配置
llm_model=ollama:qwen3.6:latest
llm_base_url=http://192.168.8.21:11434
llm_temperature=0.1
llm_max_tokens=4096

# Embedding 配置
embedding_model=nomic-embed-text:latest
embedding_base_url=http://127.0.0.1:11434

# RAG 配置
chunk_size=500
chunk_overlap=50

# 数据库
DATABASE_URL=postgresql://postgres:123456@localhost:5432/stockagent

# Tushare
TUSHARE_TOKEN=your_token_here
```

### 6.2 Pydantic Settings

```python
class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

## 7. 部署方案

### 7.1 本地开发

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 初始化数据库
python scripts/init_db.py

# 启动应用
streamlit run app.py
```

### 7.2 生产部署

```bash
# 使用 Uvicorn 部署（ASGI）
uvicorn app:app --host 0.0.0.0 --port 8501

# 或使用 Streamlit 原生部署
streamlit run app.py --server.port 8501 --server.headless true
```

---

## 8. 项目难点与解决方案

### 8.1 多 LLM 实例管理

**难点**：需要同时使用远程 LLM 和本地 Embedding。

**解决**：
- 设计了双 URL 配置（`llm_base_url` + `embedding_base_url`）
- LLM 调用使用 `ChatOllama`，Embedding 调用使用 `OllamaEmbeddings`
- 分别指向不同服务地址

### 8.2 Agent 工具调用与记忆

**难点**：需要 Agent 自主决策调用工具，同时保持多轮对话记忆。

**解决**：
- 使用 LangChain `create_agent` 的 ReAct 模式
- 基于 `MemorySaver` 实现会话级记忆
- 通过 `config={"configurable": {"thread_id": session_id}}` 隔离不同会话

### 8.3 RAG 检索质量

**难点**：确保检索结果的相关性和准确性。

**解决**：
- 采用 `RecursiveCharacterTextSplitter` 智能分块
- 配置合理的 `chunk_size=500` 和 `chunk_overlap=50`
- 使用 `nomic-embed-text` 本地 Embedding 模型
- 支持用户自定义知识库扩展

### 8.4 流式输出与 UI 交互

**难点**：Streamlit 的页面刷新机制与 Agent 流式输出冲突。

**解决**：
- 使用 Streamlit 的 `st.write_stream()` 接收生成器
- Agent 的 `stream()` 方法返回迭代器
- 图表在流式输出完成后一次性渲染

---

## 9. 项目成果

1. **功能完整性**：实现了从用户管理、数据获取、AI 分析到知识库检索的完整链路
2. **技术深度**：深入应用了 Agent、RAG、向量数据库等 AI 工程核心技术
3. **工程规范**：模块化设计、配置管理、日志系统、数据库 ORM 等工程实践
4. **数据可视化**：独立仪表盘 + AI 生成图表的双层可视化体系
5. **可扩展性**：预设配置系统、工具插件化设计，支持未来功能扩展

---

## 10. 未来规划

### 10.1 短期优化
- [ ] 仪表盘接入更多实时数据源（东方财富、新浪财经）
- [ ] 添加用户上传文档的向量化进度条
- [ ] 支持更多文档格式（Word、Markdown）
- [ ] 优化 Agent 推理速度

### 10.2 中期扩展
- [ ] 集成实时行情数据（WebSocket 推送）
- [ ] 添加多模态支持（图表理解）
- [ ] 实现 Agent 之间的协作（多 Agent 系统）

### 10.3 长期目标
- [ ] 支持多市场（美股、港股）
- [ ] 构建量化策略回测系统
- [ ] 部署为 SaaS 服务

---

## 11. 联系方式

- **GitHub**：[your-github-url]
- **邮箱**：[your-email@example.com]
- **项目演示**：[demo-url]
