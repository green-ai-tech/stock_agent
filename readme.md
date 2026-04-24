# Stock Agent - 智能股票分析系统

基于 LangChain + Streamlit 的本地化 A 股智能分析平台，集成 AI Agent 对话、RAG 知识库检索、K 线图表生成、用户管理等功能。

> 📖 **详细技术文档**：[TECHNICAL_DOCUMENT.md](./TECHNICAL_DOCUMENT.md)

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | Streamlit | 快速构建数据应用 |
| AI 框架 | LangChain (`create_agent`) | Agent 编排、工具调用、记忆管理 |
| LLM | Ollama（qwen3.6） | 本地部署大语言模型 |
| Embedding | Ollama（nomic-embed-text） | 文本向量化 |
| 向量数据库 | ChromaDB | 向量存储与相似度检索 |
| 数据源 | Tushare Pro | A 股历史行情数据 |
| 数据库 | PostgreSQL + SQLAlchemy ORM | 业务数据存储 |
| 认证 | bcrypt | 密码哈希存储 |
| 日志 | Loguru | 高性能日志框架 |
| 配置 | Pydantic Settings + `.env` | 环境变量与配置管理 |

## 核心特性

- **AI Agent 系统**：基于 ReAct 模式的智能对话，支持工具调用和多轮记忆
- **RAG 知识库**：内置投资专业知识，支持用户上传 PDF/TXT 扩展知识库
- **股票分析**：自动获取行情数据，计算技术指标，生成 K 线图/趋势图
- **用户管理**：完整的登录/注册/权限管理系统
- **会话持久化**：对话历史自动保存，支持历史回溯

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制并修改 `.env` 文件：

```env
# LLM 配置
llm_model=ollama:qwen3.6:latest
llm_base_url=http://192.168.8.21:11434
llm_temperature=0.1
llm_max_tokens=4096

# PostgreSQL 数据库
DATABASE_URL=postgresql://postgres:123456@localhost:5432/stockagent

# Tushare Token
TUSHARE_TOKEN=your_token_here
```

### 3. 初始化数据库

```bash
python scripts/init_db.py
```

默认创建管理员账号：`admin / 123456`

### 4. 启动应用

```bash
streamlit run app.py
```

## 目录结构

```
stock_agent/
├── app.py                      # Streamlit 导航入口
├── TECHNICAL_DOCUMENT.md       # 项目技术文档
├── agents/                     # AI Agent 核心
│   ├── base_agent.py           #   BaseAgent（invoke/stream/短期记忆）
│   ├── stock_agent.py          #   股票分析 Agent 工厂
│   ├── models/
│   │   ├── base_models.py      #     模型工厂（预设配置）
│   │   └── stock_models.py     #     StockAnalysisOutput 模型
│   ├── prompts/
│   │   └── system_prompt.py    #     系统提示词
│   └── tools/
│       ├── time_tools.py       #     时间工具
│       ├── stock_tools.py      #     股票数据/绘图/基础信息
│       └── rag_tools.py        #     RAG 检索工具
├── rag/                        # RAG 知识库模块
│   ├── document_loader.py      #     文档加载（PDF/TXT）
│   ├── text_splitter.py        #     文本分块
│   ├── embeddings.py           #     Ollama Embedding 封装
│   ├── vector_store.py         #     ChromaDB 向量管理
│   └── retriever.py            #     检索器
├── pages/                      # Streamlit 页面
│   ├── home.py                 #   主页（侧边栏导航）
│   ├── login.py                #   登录
│   ├── register.py             #   注册
│   ├── change_password.py      #   修改密码
│   └── admin_user.py           #   管理员用户管理
├── uis/                        # UI 组件
│   ├── ui_ai_assistant.py      #   AI 股票分析助手（对话）
│   ├── ui_stock_analysis.py    #   股市数据仪表盘（指数/K线/涨跌）
│   ├── ui_setting.py           #   模型参数设置
│   └── ui_knowledge_base.py    #   知识库管理
├── utils/                      # 工具模块
│   ├── setting.py              #   配置管理
│   ├── logger.py               #   日志
│   ├── paths.py                #   路径管理
│   ├── plot_helper.py          #   Matplotlib 样式
│   ├── db.py                   #   数据库 ORM
│   ├── auth.py                 #   认证
│   └── chat_history.py         #   会话 CRUD 服务
├── scripts/
│   └── init_db.py              #   数据库初始化
├── imgs/                       #   生成的图表
└── logs/                       #   日志文件
```

## 功能模块

### 用户系统
- 登录 / 注册 / 修改密码
- 管理员用户管理（重置密码）
- 基于 bcrypt 的密码安全存储

### 股市数据分析（仪表盘）
- 主要指数展示（上证、深证、创业板、恒生、标普500）+ 涨跌幅
- 上证指数 K 线图（近 60 交易日，含 MA5/MA20 均线）
- 市场概况：上涨/下跌/平盘家数 + 饼图分布 + 总成交额
- 手动刷新数据

### 智能股票分析助手（AI 对话）
- 输入股票代码或自然语言查询
- 自动获取日线数据 + 计算技术指标（MA5/20/60、波动率）
- 自动生成 K 线图、趋势图、成交量分布饼图
- 输出专业分析报告（技术面、量能、风险、投资建议）

### RAG 知识库
- 预置投资知识库（技术指标、风险管理等）
- 支持用户上传 PDF/TXT 文档扩展知识库
- Agent 自主决策何时调用知识库检索（RAG as Tool 模式）
- 基于 ChromaDB 的向量存储与相似度检索

### 会话管理
- 对话历史持久化存储到 PostgreSQL
- 支持创建/删除/切换多个会话
- 图表路径随消息存储，支持历史回溯

### 模型参数设置
- 运行时调整 Temperature / Max Tokens
- 支持预设切换（default / precise / creative / structure）

## 架构亮点

- **多 LLM 实例管理**：远程 LLM + 本地 Embedding 分离部署
- **ReAct Agent**：基于 LangChain 的推理-行动循环模式
- **RAG as Tool**：Agent 自主决定何时检索知识库
- **流式输出**：实时返回 Agent 推理过程
- **会话级记忆**：基于 LangGraph MemorySaver 的短期记忆

## 免责声明

本系统基于历史数据和算法模型进行分析，仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
