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
import time
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agents.models.base_models import get_chat_model
from agents.prompts import get_multi_agent_prompt
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

VALID_AGENTS = {"data_agent", "analysis_agent", "rag_agent", "FINISH"}


class RouteDecision(BaseModel):
    """Supervisor 路由决策的结构化输出，LLM 必须返回此模型"""
    next: Literal["data_agent", "analysis_agent", "rag_agent", "FINISH"] = Field(
        description="下一步应调度的 Agent 名称，如无需进一步处理则选择 FINISH"
    )


# ──────────────────────── 扩展状态 ────────────────────────

class MultiAgentState(TypedDict):
    """扩展 MessagesState，增加路由追踪字段"""
    messages: Annotated[list, "对话消息列表"]
    next_agent: str
    call_count: int
    last_agent: str


# ──────────────────────── 构建函数 ────────────────────────

def _build_supervisor_node(llm: BaseChatModel):
    """构建 Supervisor 路由节点（使用结构化输出，LLM 必须返回 RouteDecision）"""
    structured_llm = llm.with_structured_output(RouteDecision)

    def supervisor_node(state: MultiAgentState) -> dict:
        messages = state["messages"]
        call_count = state.get("call_count", 0)
        last_agent = state.get("last_agent", "")

        last_msg = messages[-1] if messages else None
        last_role = getattr(last_msg, "type", "?") if last_msg else "?"
        last_preview = (getattr(last_msg, "content", "") or "")[:80].replace("\n", " ")
        logger.info(
            f"[Supervisor] ── 路由开始 ── "
            f"消息数={len(messages)}, 上一条=[{last_role}] {last_preview}..."
        )
        logger.debug(f"[Supervisor] 上次路由: {last_agent}, 连续调用: {call_count}")

        supervisor_messages = [
            SystemMessage(content=get_multi_agent_prompt("supervisor")),
            *messages,
        ]

        t0 = time.time()
        try:
            decision: RouteDecision = structured_llm.invoke(supervisor_messages)
            next_agent = decision.next
            elapsed = time.time() - t0
            logger.info(f"[Supervisor] LLM 响应耗时: {elapsed:.2f}s, 决策: {next_agent}")
        except Exception as e:
            logger.error(f"[Supervisor] 结构化输出调用失败: {e}，默认 FINISH", exc_info=True)
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
        system_prompt=get_multi_agent_prompt("data_agent"),
        model=agent_llm,
    )

    analysis_node = _build_agent_node(
        agent_name="analysis_agent",
        tools=[plot_stock_charts],
        system_prompt=get_multi_agent_prompt("analysis_agent"),
        model=agent_llm,
    )

    rag_node = _build_agent_node(
        agent_name="rag_agent",
        tools=[search_knowledge_base],
        system_prompt=get_multi_agent_prompt("rag_agent"),
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
