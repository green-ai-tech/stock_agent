"""
多 Agent 系统 — Supervisor + 子 Agent 架构

Supervisor 使用 LLM 结构化输出决策路由，子 Agent 各司其职：
  - data_agent      : 获取股票行情数据和基础信息
  - analysis_agent  : 生成图表和技术分析报告
  - rag_agent       : 从知识库检索专业知识

通过 settings.use_multi_agent 开关切换，与单 Agent 共享同一 invoke 接口。
"""

from __future__ import annotations

import json
import re
import time
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agents.models.base_models import get_chat_model
from agents.tools.rag_tools import search_knowledge_base
from agents.tools.stock_tools import (
    get_stock_basic_info,
    get_stock_daily_data,
    plot_stock_charts,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────── 常量 ────────────────────────────

MAX_CONSECUTIVE_CALLS = 3

AGENT_NAMES = Literal["data_agent", "analysis_agent", "rag_agent", "FINISH"]

VALID_AGENTS = {"data_agent", "analysis_agent", "rag_agent", "FINISH"}


# ──────────────────────────── Supervisor ────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """\
你是一个任务路由器，负责将用户请求分配给最合适的专业 Agent。

## 可用 Agent

- **data_agent**：获取股票行情数据（日线、K 线）和基础信息（行业、市值、PE 等）。
  当用户需要查看股票数据、获取价格走势、查询基本面信息时选择。

- **analysis_agent**：生成股票分析图表（K 线图、趋势图、饼图）并提供技术分析报告。
  当用户需要图表、技术指标分析、趋势研判时选择。

- **rag_agent**：从投资知识库中检索专业知识。
  当用户询问投资概念、技术指标含义（如 MACD、KDJ）、财务分析方法、K 线形态等知识性问题时选择。

## 路由规则

1. 根据用户问题的**核心意图**选择最合适的 Agent
2. 如果前一个 Agent 的输出已经完整回答了用户问题，输出 FINISH
3. 不要连续重复调用同一个 Agent（系统会强制限制）
4. 如果用户只是打招呼、闲聊或问题与股票无关，直接输出 FINISH
5. 复杂任务可以分步调用多个 Agent（先获取数据 → 再分析 → 再检索补充知识）

## 输出格式

你必须严格按以下 JSON 格式输出，不要输出任何其他内容：
{"next": "data_agent"}
或
{"next": "analysis_agent"}
或
{"next": "rag_agent"}
或
{"next": "FINISH"}

## 当前对话

请根据对话中最新的用户消息做出路由决策。"""


# ──────────────────────── 扩展状态 ────────────────────────

class MultiAgentState(TypedDict):
    """扩展 MessagesState，增加路由追踪字段"""
    messages: Annotated[list, "对话消息列表"]
    next_agent: str
    call_count: int
    last_agent: str


# ──────────────────────── 子 Agent 提示词 ────────────────────────

DATA_AGENT_PROMPT = """\
你是股票数据专家。你的职责是获取股票行情数据和基础信息。

可用工具：
1. get_stock_daily_data — 获取股票历史日线数据（含 MA5/20/60、波动率等技术指标）
2. get_stock_basic_info — 获取股票基础信息（名称、行业、市值、PE、PB 等）

工作流程：
- 根据用户输入的股票代码，先获取基础信息确认股票身份
- 再获取日线数据提供行情概览
- 将数据以清晰的格式呈现给用户

注意：股票代码格式如 600519.SH（上海）、000858.SZ（深圳）。如果用户只说股票名称，请先确认代码。"""

ANALYSIS_AGENT_PROMPT = """\
你是股票技术分析专家。你的职责是生成分析图表并提供技术分析报告。

可用工具：
1. plot_stock_charts — 绘制股票分析图表（K 线图、趋势图、成交量饼图），返回图表文件路径

工作流程：
- 根据用户输入的股票代码生成分析图表
- 基于图表数据提供专业的技术分析
- 分析维度：均线系统、量价配合、趋势方向、风险信号

注意：如果数据缓存不存在，工具会自动获取数据。请确保输出包含图表路径供前端展示。"""

RAG_AGENT_PROMPT = """\
你是投资知识库检索专家。你的职责是从知识库中检索相关专业知识并回答用户问题。

可用工具：
1. search_knowledge_base — 搜索投资知识库，获取专业知识参考内容

使用场景：
- 用户询问投资概念（如"什么是 MACD"）
- 用户询问技术指标含义和用法
- 用户询问财务分析方法
- 用户询问 K 线形态定义

工作流程：
- 根据用户问题构造合适的搜索关键词
- 调用知识库检索
- 基于检索结果提供准确、专业的回答
- 如果检索结果不足，结合自身知识补充说明"""


# ──────────────────────── 构建函数 ────────────────────────

def _parse_supervisor_response(text: str) -> str:
    """从 LLM 响应中解析路由决策（兼容 JSON 和纯文本）"""
    # 尝试解析 JSON
    json_match = re.search(r'\{[^}]*"next"\s*:\s*"[^"]*"[^}]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            candidate = data.get("next", "FINISH")
            if candidate in VALID_AGENTS:
                return candidate
        except json.JSONDecodeError:
            pass

    # 回退：关键词匹配
    text_lower = text.lower()
    for agent_name in ["data_agent", "analysis_agent", "rag_agent"]:
        if agent_name in text_lower:
            return agent_name
    if "finish" in text_lower:
        return "FINISH"

    return "FINISH"


def _build_supervisor_node(llm: BaseChatModel):
    """构建 Supervisor 路由节点（使用普通 LLM 调用 + JSON 解析）"""

    def supervisor_node(state: MultiAgentState) -> dict:
        messages = state["messages"]
        call_count = state.get("call_count", 0)
        last_agent = state.get("last_agent", "")

        # 取最后一条用户/AI 消息摘要
        last_msg = messages[-1] if messages else None
        last_role = getattr(last_msg, "type", "?") if last_msg else "?"
        last_preview = (getattr(last_msg, "content", "") or "")[:80].replace("\n", " ")
        logger.info(
            f"[Supervisor] ── 路由开始 ── "
            f"消息数={len(messages)}, 上一条=[{last_role}] {last_preview}..."
        )
        logger.debug(f"[Supervisor] 上次路由: {last_agent}, 连续调用: {call_count}")

        # 构造给 Supervisor 的消息
        supervisor_messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            *messages,
        ]

        t0 = time.time()
        try:
            response = llm.invoke(supervisor_messages)
            response_text = response.content if hasattr(response, "content") else str(response)
            elapsed = time.time() - t0

            logger.info(f"[Supervisor] LLM 响应耗时: {elapsed:.2f}s")
            logger.debug(f"[Supervisor] LLM 原始输出: {response_text[:200]}")

            next_agent = _parse_supervisor_response(response_text)
            logger.info(f"[Supervisor] ── 路由决策: {next_agent} ──")
        except Exception as e:
            logger.error(f"[Supervisor] 路由决策失败: {e}，默认 FINISH", exc_info=True)
            next_agent = "FINISH"

        # 连续调用计数
        if next_agent == last_agent:
            call_count += 1
        else:
            call_count = 1

        if call_count >= MAX_CONSECUTIVE_CALLS:
            logger.warning(
                f"[Supervisor] ⚠️ 连续调用 {next_agent} 达 {call_count} 次，强制 FINISH"
            )
            next_agent = "FINISH"

        return {
            "next_agent": next_agent,
            "call_count": call_count,
            "last_agent": next_agent if next_agent != "FINISH" else last_agent,
        }

    return supervisor_node


def _build_agent_node(
    agent_name: str,
    tools: list,
    system_prompt: str,
    model: BaseChatModel | None = None,
):
    """构建子 Agent 节点（使用 create_agent 轻量封装）"""
    if model is None:
        model = get_chat_model(temperature=0.3, max_tokens=4096)

    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    def agent_node(state: MultiAgentState) -> dict:
        messages = state["messages"]
        logger.info(f"[{agent_name}] ═══ 开始执行 ═══ 消息数: {len(messages)}")

        try:
            t0 = time.time()
            result = agent_graph.invoke({"messages": messages})
            elapsed = time.time() - t0

            result_messages = result.get("messages", [])
            logger.info(f"[{agent_name}] 执行耗时: {elapsed:.2f}s, 输出消息数: {len(result_messages)}")

            # 日志：遍历工具调用记录
            tool_calls_logged = 0
            for msg in result_messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_logged += 1
                        tc_name = tc.get("name", "?")
                        tc_args = tc.get("args", {})
                        logger.info(
                            f"[{agent_name}] 🔧 工具调用 #{tool_calls_logged}: "
                            f"{tc_name}({json.dumps(tc_args, ensure_ascii=False)})"
                        )
                if isinstance(msg, ToolMessage):
                    tool_content = (msg.content or "")[:150].replace("\n", " ")
                    logger.debug(f"[{agent_name}] 📥 工具返回: {tool_content}...")

            # 提取最后一条 AI 消息
            ai_message = None
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_message = msg
                    break

            if ai_message:
                output_preview = ai_message.content[:120].replace("\n", " ")
                logger.info(
                    f"[{agent_name}] ✅ 执行完成, "
                    f"输出 {len(ai_message.content)} 字符: {output_preview}..."
                )
                return {"messages": [ai_message]}
            else:
                logger.warning(f"[{agent_name}] ⚠️ 未产生有效输出")
                return {
                    "messages": [
                        AIMessage(content=f"[{agent_name}] 未产生有效输出，请重试。")
                    ]
                }

        except Exception as e:
            logger.error(f"[{agent_name}] ❌ 执行失败: {e}", exc_info=True)
            return {
                "messages": [
                    AIMessage(content=f"[{agent_name}] 执行出错: {str(e)}")
                ]
            }

    return agent_node


def _route_next(state: MultiAgentState) -> str:
    """条件边：根据 Supervisor 决策路由到下一个节点"""
    next_agent = state.get("next_agent", "FINISH")

    if next_agent == "FINISH":
        return END

    # 映射到实际节点名
    node_map = {
        "data_agent": "data_agent",
        "analysis_agent": "analysis_agent",
        "rag_agent": "rag_agent",
    }

    target = node_map.get(next_agent)
    if target is None:
        logger.warning(f"[Router] 未知的 Agent: {next_agent}，回退到 FINISH")
        return END

    return target


# ──────────────────────── 图构建 ────────────────────────

def create_multi_agent_graph() -> StateGraph:
    """
    创建并编译多 Agent 图。

    Returns:
        CompiledStateGraph，支持 .invoke({"messages": [...]}) 和 .stream()
    """
    logger.info("正在构建多 Agent 系统 (Supervisor + 3 子 Agent)...")

    # 获取 Supervisor 专用 LLM（temperature=0 确保路由稳定）
    supervisor_llm = get_chat_model(temperature=0, max_tokens=256)

    # 获取子 Agent 共享 LLM
    agent_llm = get_chat_model(temperature=0.3, max_tokens=4096)

    # 构建节点
    supervisor_node = _build_supervisor_node(supervisor_llm)

    data_node = _build_agent_node(
        agent_name="data_agent",
        tools=[get_stock_daily_data, get_stock_basic_info],
        system_prompt=DATA_AGENT_PROMPT,
        model=agent_llm,
    )

    analysis_node = _build_agent_node(
        agent_name="analysis_agent",
        tools=[plot_stock_charts],
        system_prompt=ANALYSIS_AGENT_PROMPT,
        model=agent_llm,
    )

    rag_node = _build_agent_node(
        agent_name="rag_agent",
        tools=[search_knowledge_base],
        system_prompt=RAG_AGENT_PROMPT,
        model=agent_llm,
    )

    # 构建图
    graph = StateGraph(MultiAgentState)

    # 添加节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_agent", data_node)
    graph.add_node("analysis_agent", analysis_node)
    graph.add_node("rag_agent", rag_node)

    # 添加边
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_next,
        {
            "data_agent": "data_agent",
            "analysis_agent": "analysis_agent",
            "rag_agent": "rag_agent",
            END: END,
        },
    )
    graph.add_edge("data_agent", "supervisor")
    graph.add_edge("analysis_agent", "supervisor")
    graph.add_edge("rag_agent", "supervisor")

    # 编译
    compiled = graph.compile()
    logger.success("多 Agent 系统构建完成")
    logger.info("  ├─ Supervisor (路由决策)")
    logger.info("  ├─ data_agent (股票数据)")
    logger.info("  ├─ analysis_agent (图表分析)")
    logger.info("  └─ rag_agent (知识库检索)")

    return compiled
