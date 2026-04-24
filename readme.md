# Stock Agent - 智能股票分析系统

基于 LangChain + Streamlit 的本地化 A 股智能分析平台，集成 AI Agent 对话、K 线图表生成、用户管理等功能。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Streamlit |
| AI 框架 | LangChain (`create_agent`) |
| LLM | Ollama（本地部署，默认 qwen3.6:latest） |
| 数据源 | Tushare Pro |
| 数据库 | PostgreSQL + SQLAlchemy ORM |
| 认证 | bcrypt 密码哈希 |
| 日志 | Loguru |
| 配置 | Pydantic Settings + `.env` |

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
│       └── stock_tools.py      #     股票数据/绘图/基础信息
├── pages/                      # Streamlit 页面
│   ├── home.py                 #   主页（侧边栏导航）
│   ├── login.py                #   登录
│   ├── register.py             #   注册
│   ├── change_password.py      #   修改密码
│   └── admin_user.py           #   管理员用户管理
├── uis/                        # UI 组件
│   ├── ui_ai_assistant.py      #   AI 股票分析助手
│   ├── ui_stock_analysis.py    #   通用对话（流式输出）
│   └── ui_setting.py           #   模型参数设置
├── utils/                      # 工具模块
│   ├── setting.py              #   配置管理
│   ├── logger.py               #   日志
│   ├── paths.py                #   路径管理
│   ├── plot_helper.py          #   Matplotlib 样式
│   ├── db.py                   #   数据库 ORM
│   └── auth.py                 #   认证
├── scripts/
│   └── init_db.py              #   数据库初始化
├── imgs/                       #   生成的图表
└── logs/                       #   日志文件
```

## 功能模块

### 用户系统
- 登录 / 注册 / 修改密码
- 管理员用户管理（重置密码）

### 股市数据分析
- 通用 AI 对话（流式输出）

### 智能股票分析助手
- 输入股票代码或自然语言查询
- 自动获取日线数据 + 计算技术指标（MA5/20/60、波动率）
- 自动生成 K 线图、趋势图、成交量分布饼图
- 输出专业分析报告（技术面、量能、风险、投资建议）

### 模型参数设置
- 运行时调整 Temperature / Max Tokens
- 支持预设切换（default / precise / creative / structure）

## 免责声明

本系统基于历史数据和算法模型进行分析，仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
