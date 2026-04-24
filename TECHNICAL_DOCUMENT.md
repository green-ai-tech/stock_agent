# Stock Agent — AI Agent 系统技术文档

> **定位**：面向大模型 / Agent 开发岗位的技术深度展示  
> **核心框架**：LangChain + LangGraph  
> **架构模式**：Supervisor + 子 Agent 多 Agent 系统

---

## 1. 项目概述

基于 LangChain + LangGraph 的 A 股智能分析平台，核心是 **多 Agent 系统**：Supervisor 路由 + 3 个专业子 Agent（数据获取、图表分析、知识库检索）协作完成任务。支持单 Agent（ReAct）和多 Agent（Supervisor）两种模式通过环境变量切换。

---

## 2. Agent 设计流程 + 多 Agent 架构（核心）

### 2.1 整体架构

```
用户输入
    │
    ▼
┌──────────────┐
│  Supervisor  │ ← LLM 路由决策（temperature=0）
│  (Router)    │   输出 JSON: {"next": "data_agent"|"analysis_agent"|"rag_agent"|"FINISH"}
└──────┬───────┘
       │
       ├────────────────────┬───────────────────┐
       ▼                    ▼                   ▼
┌──────────────┐   ┌────────────────┐   ┌──────────────┐
│  data_agent  │   │ analysis_agent │   │  rag_agent   │
│  (ReAct)     │   │   (ReAct)      │   │   (ReAct)    │
│  工具:       │   │   工具:        │   │   工具:      │
│  股票数据API │   │   图表生成     │   │   向量检索   │
└──────────────┘   └────────────────┘   └──────────────┘
       │                    │                   │
       └────────────────────┼───────────────────┘
                            ▼
                       Supervisor
                      (循环或 FINISH)
                            │
                            ▼
                        最终回复
```

**两种模式**（通过 `USE_MULTI_AGENT` 环境变量切换）：
- **单 Agent 模式**：一个 ReAct Agent 集成所有工具，自主决策
- **多 Agent 模式**：Supervisor 路由 + 专业子 Agent 分工协作

### 2.2 完整调用时序图

```
用户: "分析一下贵州茅台600519的技术走势"
    │
    ▼
Supervisor (LLM)
    │  System Prompt 描述各 Agent 能力 + 输出格式
    │  LLM 分析用户意图 → {"next": "data_agent"}
    │
    ▼
data_agent (create_agent)
    │  工具: get_stock_daily_data, get_stock_basic_info
    │  ReAct 循环:
    │    Thought: 需要获取股票数据
    │    Action: get_stock_daily_data(ts_code="600519.SH", days=120)
    │    Observation: 收盘价 1850.00, MA5=1845.20, 波动率=1.8%...
    │    Thought: 需要获取基础信息
    │    Action: get_stock_basic_info(ts_code="600519.SH")
    │    Observation: 贵州茅台, 白酒行业, PE=28.5...
    │    Thought: 数据已齐全
    │    Final Answer: [整合数据输出]
    │
    ▼
Supervisor (LLM)
    │  看到 data_agent 的输出
    │  决策: {"next": "analysis_agent"}
    │
    ▼
analysis_agent (create_agent)
    │  工具: plot_stock_charts
    │  ReAct 循环:
    │    Thought: 需要生成图表
    │    Action: plot_stock_charts(ts_code="600519.SH")
    │    Observation: 图表已生成: K线图: /path/kline.png, 趋势图: /path/trend.png
    │    Final Answer: [技术分析报告 + 图表路径]
    │
    ▼
Supervisor (LLM)
    │  看到完整结果（数据+图表+分析）
    │  决策: {"next": "FINISH"}
    │
    ▼
最终输出: 数据概览 + 技术分析 + 图表
```

### 2.3 Supervisor 路由实现

Supervisor 不使用 ReAct 工具调用，而是 **纯 LLM 决策节点**（减少推理开销）：

```python
SUPERVISOR_SYSTEM_PROMPT = """\
你是一个任务路由器，负责将用户请求分配给最合适的专业 Agent。

可用 Agent：
- data_agent：获取股票行情数据和基础信息
- analysis_agent：生成图表和技术分析报告
- rag_agent：从知识库检索专业知识

规则：
1. 根据用户问题的核心意图选择最合适的 Agent
2. 如果前一个 Agent 的输出已完整回答问题，输出 FINISH
3. 不要连续重复调用同一个 Agent

输出格式（严格 JSON）：
{"next": "data_agent"} 或 {"next": "analysis_agent"} 或 {"next": "rag_agent"} 或 {"next": "FINISH"}
"""

def _build_supervisor_node(llm):
    def supervisor_node(state: MultiAgentState) -> dict:
        messages = state["messages"]
        call_count = state.get("call_count", 0)
        last_agent = state.get("last_agent", "")

        # LLM 路由决策
        response = llm.invoke([SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT), *messages])
        next_agent = _parse_supervisor_response(response.content)

        # 连续调用限制（防死循环）
        if next_agent == last_agent:
            call_count += 1
        else:
            call_count = 1
        if call_count >= MAX_CONSECUTIVE_CALLS:  # 3 次上限
            next_agent = "FINISH"

        return {"next_agent": next_agent, "call_count": call_count, "last_agent": next_agent}
    return supervisor_node
```

**Supervisor 输出解析**（兼容 JSON 和纯文本，提升鲁棒性）：

```python
def _parse_supervisor_response(text: str) -> str:
    """优先解析 JSON，回退到关键词匹配"""
    json_match = re.search(r'\{[^}]*"next"\s*:\s*"[^"]*"[^}]*\}', text)
    if json_match:
        data = json.loads(json_match.group())
        candidate = data.get("next", "FINISH")
        if candidate in VALID_AGENTS:
            return candidate

    # 回退：关键词匹配
    for agent_name in ["data_agent", "analysis_agent", "rag_agent"]:
        if agent_name in text.lower():
            return agent_name
    return "FINISH"
```

### 2.4 子 Agent 创建方式

子 Agent 使用 `langchain.agents.create_agent()` 轻量创建（不使用 InMemorySaver，因为 Supervisor 管理全局状态）：

```python
def _build_agent_node(agent_name: str, tools: list, system_prompt: str, model):
    """子 Agent 节点工厂"""
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    def agent_node(state: MultiAgentState) -> dict:
        messages = state["messages"]
        result = agent_graph.invoke({"messages": messages})

        # 提取最后一条 AI 消息
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                return {"messages": [msg]}
        return {"messages": [AIMessage(content="未产生有效输出")]}
    return agent_node
```

### 2.5 子 Agent 职责边界与工具分配

| 子 Agent | 职责 | 工具 | 提示词要点 |
|----------|------|------|-----------|
| `data_agent` | 获取股票行情 + 基本面 | `get_stock_daily_data`, `get_stock_basic_info` | 强调数据准确性，要求先获取基础信息确认股票身份 |
| `analysis_agent` | 图表生成 + 技术分析 | `plot_stock_charts` | 强调输出图表路径，提供多维度技术分析（均线、量价、趋势） |
| `rag_agent` | 知识库检索 + 专业解答 | `search_knowledge_base` | 强调基于检索结果回答，避免幻觉 |

### 2.6 LangGraph 图构建

```python
def create_multi_agent_graph():
    graph = StateGraph(MultiAgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_agent", data_node)
    graph.add_node("analysis_agent", analysis_node)
    graph.add_node("rag_agent", rag_node)

    # 边定义
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", _route_next, {
        "data_agent": "data_agent",
        "analysis_agent": "analysis_agent",
        "rag_agent": "rag_agent",
        END: END,
    })
    # 所有子 Agent 执行完回到 Supervisor
    graph.add_edge("data_agent", "supervisor")
    graph.add_edge("analysis_agent", "supervisor")
    graph.add_edge("rag_agent", "supervisor")

    return graph.compile()
```

---

## 3. 工具调用机制与稳定性优化

### 3.1 工具定义方式

使用 LangChain `@tool` 装饰器定义工具，每个工具自带文档字符串（Agent 通过 description 决定何时调用）：

```python
@tool
def get_stock_daily_data(
    ts_code: Annotated[str, "股票代码，如 '600519.SH'"],
    days: Annotated[int, "获取数据的天数，默认120"] = 120
) -> str:
    """获取股票日线数据，返回包含技术指标的数据摘要。"""
    try:
        df = pro.daily(ts_code=ts_code, ...)
        # 计算 MA5/20/60、波动率等技术指标
        summary = format_summary(df)
        return summary
    except Exception as e:
        return f"获取数据失败: {str(e)}"
```

**设计要点**：
- `Annotated[str, "描述"]` 提供参数说明，LLM 通过 description 理解参数含义
- 返回值为纯文本摘要（便于 LLM 理解和整合）
- 异常捕获返回错误信息而非抛出异常（避免中断 Agent 循环）

### 3.2 工具返回数据结构

| 工具 | 返回格式 | 设计考量 |
|------|----------|----------|
| `get_stock_daily_data` | 纯文本摘要（含表格） | LLM 更易理解和整合文本数据 |
| `get_stock_basic_info` | 纯文本（键值对） | 结构化信息便于引用 |
| `plot_stock_charts` | 文本 + 文件路径 | 路径供前端解析展示图片 |
| `search_knowledge_base` | 检索结果拼接文本 | 直接作为上下文供 LLM 参考 |

```python
# plot_stock_charts 返回格式示例
return f"""图表已生成:
- K线图: /path/to/kline_600519_20260424.png
- 趋势图: /path/to/trend_600519_20260424.png
- 饼图: /path/to/pie_600519_20260424.png
"""
```

### 3.3 防止 Agent 陷入重复调用循环

**三层防护机制**：

**1. 提示词约束**（软约束）
```
不要连续重复调用同一个 Agent。
```

**2. 连续调用计数器**（硬约束）
```python
MAX_CONSECUTIVE_CALLS = 3

if next_agent == last_agent:
    call_count += 1
else:
    call_count = 1
if call_count >= MAX_CONSECUTIVE_CALLS:
    next_agent = "FINISH"  # 强制结束
```

**3. LangGraph 递归限制**（底层保护）
```python
config = {"recursion_limit": 10}  # 最多 10 轮循环
result = graph.invoke(input, config=config)
```

### 3.4 工具调用失败降级

子 Agent 内部的工具调用失败由 Agent 自身处理（ReAct 模式自动重试或换策略）：
```python
try:
    result = agent_graph.invoke({"messages": messages})
except Exception as e:
    # 返回错误消息而非中断整个流程
    return {"messages": [AIMessage(content=f"[{agent_name}] 执行出错: {str(e)}")]}
```

---

## 4. 记忆管理

### 4.1 短期记忆：基于 LangGraph Checkpointer

**单 Agent 模式**：BaseAgent 使用 `InMemorySaver` 实现会话级短期记忆：

```python
checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer,  # 注入短期记忆
)

# 通过 thread_id 隔离不同会话
config = {"configurable": {"thread_id": "user_123_conv_456"}}
result = agent.invoke(input, config=config)
```

**关键机制**：
- `InMemorySaver` 基于 LangGraph `MemorySaver`，在内存中存储对话状态
- `thread_id` 是会话隔离的关键——不同 thread_id 对应完全独立的对话上下文
- 用户切换会话时，前端传入不同的 `thread_id`，Agent 自动加载对应历史

**多 Agent 模式**：
- Supervisor 的 `MultiAgentState.messages` 包含完整对话历史（全局共享）
- 子 Agent 不使用独立 checkpointer（避免状态碎片化）
- 每个子 Agent 接收全局消息列表，处理后追加结果，实现 Agent 间上下文传递

### 4.2 多 Agent 间记忆共享机制

```
state = MultiAgentState(
    messages=[HumanMessage("分析茅台")],  # 全局共享
    next_agent="data_agent",
    call_count=1,
    last_agent="data_agent"
)

# data_agent 执行后 → 追加 AIMessage 到 messages
state["messages"] = [HumanMessage("分析茅台"), AIMessage("茅台数据: ...")]

# Supervisor 看到完整历史（包括 data_agent 的输出）
# → 决策: analysis_agent

# analysis_agent 也看到完整历史
# → 基于 data_agent 的数据生成图表和分析
```

### 4.3 会话持久化（业务层）

除 LangGraph 的短期记忆外，业务层还将对话持久化到 PostgreSQL：

```python
# 消息存储时序
user_input → Agent.invoke() → 持久化 user Message
                             → Agent 处理
                             → 持久化 AI Message + chart_paths
```

数据模型：
```python
class Message(Base):
    id: Integer (PK)
    conversation_id: Integer (FK)
    role: String ('human' / 'ai')
    content: Text
    chart_paths: JSON  # 图表路径列表
    created_at: DateTime
```

---

## 5. RAG 集成

### 5.1 架构：RAG as Tool

Agent 自主决定何时调用知识库，而非每次查询都检索：

```
用户: "什么是 MACD？"
    │
    ▼
Agent 推理: 用户在问投资概念 → 调用 search_knowledge_base
    │
    ▼
ChromaDB 向量检索 → 返回相关文档片段
    │
    ▼
Agent 整合检索结果 + 自身知识 → 生成回答
```

### 5.2 向量化流程

```
PDF/TXT 文档
    │
    ▼
pypdf 解析 → 纯文本
    │
    ▼
RecursiveCharacterTextSplitter
    chunk_size=500, chunk_overlap=50
    │
    ▼
OllamaEmbeddings (nomic-embed-text)
    │
    ▼
ChromaDB 持久化存储
    collection: "preset" (预置知识) / "user_{id}" (用户知识)
```

### 5.3 RAG 检索工具实现

```python
@tool
def search_knowledge_base(
    query: Annotated[str, "搜索查询"],
    top_k: Annotated[int, "返回结果数量"] = 3,
) -> str:
    """搜索投资知识库"""
    collection = get_collection("preset")
    results = collection.query(query_texts=[query], n_results=top_k)
    return "\n\n".join(results["documents"][0])
```

**设计要点**：
- `@tool` docstring 描述使用场景，Agent 通过描述判断何时调用
- 返回纯文本拼接结果，直接作为 LLM 上下文
- 预置知识库和用户知识库集合隔离

### 5.4 防止幻觉

- **RAG as Tool 模式**：Agent 仅在需要时调用检索，而非所有查询都走 RAG
- **提示词约束**：要求子 Agent 基于检索结果回答，避免编造
- **检索结果注入**：检索到的原文片段直接作为上下文，不经过 LLM 改写

---

## 6. 单 Agent vs 多 Agent 对比

| 维度 | 单 Agent (ReAct) | 多 Agent (Supervisor) |
|------|------------------|----------------------|
| **架构** | 一个 Agent 集成所有工具 | Supervisor 路由 + 专业子 Agent |
| **工具选择** | LLM 自主决策 | Supervisor 路由 + 子 Agent 内部决策 |
| **职责边界** | 模糊（一个 Agent 做所有事） | 清晰（数据/分析/知识各司其职） |
| **可控性** | 低（无法控制调用顺序） | 高（Supervisor 控制执行流程） |
| **调试难度** | 高（所有工具混在一起） | 低（每步路由可追踪） |
| **性能** | 单次推理可能需多次工具调用 | 分步推理，每步专注单一任务 |
| **扩展性** | 工具增多后 prompt 复杂度爆炸 | 新增 Agent 即可，不影响现有结构 |

### 切换机制

```env
USE_MULTI_AGENT=true   # Supervisor + 子 Agent 模式
USE_MULTI_AGENT=false  # 单 Agent ReAct 模式（默认）
```

UI 层通过统一接口调用，无需修改代码：
```python
if settings.use_multi_agent:
    agent = create_multi_agent_graph()    # CompiledStateGraph
else:
    agent = create_stock_analyst_agent()  # CompiledStateGraph

# 两种模式返回相同的 invoke 接口
result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
```

---

## 7. 流式输出

BaseAgent 封装了 LangGraph 的 `stream()` 方法，支持三种模式：

```python
def stream(self, input_text, stream_mode="messages"):
    for chunk in self.graph.stream(input, stream_mode=stream_mode, config=config):
        if stream_mode == "messages":
            # chunk 是 (message, metadata) 元组
            message, metadata = chunk
            if isinstance(message, AIMessage) and message.content:
                yield message.content
        elif stream_mode == "updates":
            # chunk 是状态更新字典
            if "messages" in chunk:
                yield chunk["messages"][-1].content
```

多 Agent 模式通过 `stream_mode="updates"` 追踪路由过程：
```python
for event in agent.stream(input, stream_mode="updates"):
    if "supervisor" in event:
        decision = event["supervisor"].get("next_agent", "")
        # UI 显示: "正在调用 data_agent..."
```

---

## 8. 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **Agent 框架** | LangChain + LangGraph | Agent 编排、工具调用、状态图 |
| **LLM** | Ollama (qwen3.6) | 本地部署大语言模型 |
| **Embedding** | Ollama (nomic-embed-text) | 文本向量化 |
| **向量数据库** | ChromaDB | 向量存储与相似度检索 |
| **数据源** | Tushare Pro | A 股历史行情数据 |
| **关系数据库** | PostgreSQL + SQLAlchemy | 会话持久化 |
| **前端** | Streamlit | Web UI |
| **图表** | Matplotlib | K 线图、趋势图生成 |
| **日志** | Loguru | 高性能日志 |
| **配置** | Pydantic Settings | 环境变量管理 |

---

## 9. 项目结构

```
stock_agent/
├── agents/
│   ├── base_agent.py           # BaseAgent 封装（invoke/stream/记忆）
│   ├── stock_agent.py          # 单 Agent 工厂（ReAct 模式）
│   ├── multi_agent.py          # 多 Agent 图（Supervisor + 子 Agent）
│   ├── models/base_models.py   # LLM 模型工厂（预设配置）
│   ├── prompts/system_prompt.py # 提示词系统
│   └── tools/
│       ├── stock_tools.py      # 股票数据/绘图工具
│       ├── rag_tools.py        # RAG 检索工具
│       └── time_tools.py       # 时间工具
├── rag/
│   ├── document_loader.py      # 文档加载
│   ├── text_splitter.py        # 文本分块
│   ├── embeddings.py           # Embedding 封装
│   ├── vector_store.py         # ChromaDB 管理
│   └── retriever.py            # 检索器
├── uis/                        # UI 组件
├── pages/                      # Streamlit 页面
├── utils/                      # 配置/日志/数据库
└── scripts/init_db.py          # 数据库初始化
```

---

## 10. 部署

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化数据库
python scripts/init_db.py

# 启动
streamlit run app.py
```

---

**联系方式**
- GitHub: [your-github-url]
- 邮箱: [your-email@example.com]
