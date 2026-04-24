# uis/ui_ai_assistant.py
import streamlit as st
import re
from pathlib import Path
from datetime import datetime
from utils.logger import logger
from agents.stock_agent import create_stock_analyst_agent
from utils.setting import settings
from utils.paths import get_stock_charts_dir
from utils.chat_history import add_message, create_conversation, generate_title


def apply_chart_styling():
    st.markdown("""
    <style>
    .chart-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 0.8rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .chart-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.12);
    }
    .chart-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-align: center;
        color: #1f2937;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def _ensure_conversation():
    """确保当前有会话，没有则自动创建"""
    if st.session_state.get("conv_id"):
        return
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    conv = create_conversation(user_id=user_id, agent_type="stock")
    if conv:
        st.session_state.conv_id = conv.id


def ui_ai_assistant():
    apply_chart_styling()

    if "stock_messages" not in st.session_state:
        st.session_state.stock_messages = []
    if "stock_agent" not in st.session_state:
        with st.spinner("正在初始化智能分析师..."):
            try:
                st.session_state.stock_agent = create_stock_analyst_agent()
                logger.info("股票分析智能体加载成功")
                st.success("✅ 智能体初始化成功")
            except Exception as e:
                logger.error(f"智能体加载失败: {e}", exc_info=True)
                st.error(f"❌ 智能体初始化失败：{str(e)}")
                return

    st.title("📊 智能股票分析助手")
    st.caption(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 模型: {settings.llm_model}")

    with st.expander("📌 使用说明", expanded=False):
        st.markdown("""
        1. 输入股票代码（如 `600519.SH` 或 `000858.SZ`）
        2. 也可以直接提问，例如：
           - "分析一下贵州茅台"
           - "600519 的技术走势"
        3. 系统会自动调用工具获取数据、生成图表并输出专业分析报告
        """)
        st.markdown("⚠️ **免责声明**：本分析基于历史数据和算法模型，不构成投资建议。投资有风险，决策需谨慎。")

    # 显示历史消息
    for msg in st.session_state.stock_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if "charts" in msg["content"] and msg["content"]["charts"]:
                    st.subheader("📊 技术图表")
                    for title, path in msg["content"]["charts"].items():
                        with st.container():
                            st.markdown(f'<div class="chart-card">', unsafe_allow_html=True)
                            st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
                            st.image(path, width='stretch')
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(msg["content"]["text"])
            else:
                st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("请输入股票代码或分析问题..."):
        _ensure_conversation()

        # 显示用户消息
        st.session_state.stock_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 持久化用户消息
        if st.session_state.get("conv_id"):
            if len(st.session_state.stock_messages) == 1:
                from utils.chat_history import rename_conversation
                rename_conversation(st.session_state.conv_id, generate_title(prompt))
            add_message(st.session_state.conv_id, "user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在分析中，请稍候..."):
                try:
                    agent = st.session_state.stock_agent
                    response = agent.invoke({
                        "messages": [{"role": "user", "content": prompt}]
                    })

                    messages = response.get("messages", [])

                    # 提取 AI 最终回复
                    final_text = ""
                    for msg in reversed(messages):
                        if hasattr(msg, "content") and msg.content and getattr(msg, "type", "") == "ai":
                            final_text = msg.content
                            break

                    # 提取图表路径
                    chart_paths = {}
                    tool_message_found = False
                    for msg in messages:
                        if hasattr(msg, "type") and msg.type == "tool":
                            tool_message_found = True
                            msg_content = msg.content
                            if "图表已生成" in msg_content:
                                lines = msg_content.split("\n")
                                for line in lines:
                                    if line.strip().startswith("-") and ": " in line:
                                        parts = line.split(": ", 1)
                                        if len(parts) == 2:
                                            chart_type = parts[0].strip("- ").strip()
                                            file_path = parts[1].strip()
                                            full_path = Path(file_path)
                                            if full_path.exists():
                                                chart_paths[chart_type] = str(full_path)
                                            else:
                                                logger.warning(f"图表文件不存在: {full_path}")

                    # 备用正则
                    if not chart_paths and final_text:
                        pattern = r'(kline|trend|pie)_[^_\s]+_\d{8}_\d{6}\.png'
                        matches = re.findall(pattern, final_text)
                        if matches:
                            type_names = {"kline": "K线图", "trend": "趋势图", "pie": "饼图"}
                            for match in matches:
                                chart_type_key = match.split('_')[0]
                                chart_title = type_names.get(chart_type_key, "图表")
                                full_path = get_stock_charts_dir() / match
                                if full_path.exists():
                                    chart_paths[chart_title] = str(full_path)

                    if not chart_paths and tool_message_found:
                        st.info("📉 工具已调用，但未能解析出图表路径。请检查调试信息。")
                    elif not chart_paths:
                        st.info("📉 本次分析未生成图表（可能数据不足或模型未调用绘图工具）。")

                    # 显示图表
                    if chart_paths:
                        st.subheader("📊 技术图表")
                        for title, path in chart_paths.items():
                            with st.container():
                                st.markdown(f'<div class="chart-card">', unsafe_allow_html=True)
                                st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
                                st.image(path, width='stretch')
                                st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

                    # 显示分析文本
                    if final_text:
                        st.markdown(final_text)
                    else:
                        st.warning("模型未返回任何内容，请检查后端日志")
                        final_text = "无响应"

                    assistant_content = {
                        "text": final_text,
                        "charts": chart_paths,
                        "metrics": {},
                    }
                    st.session_state.stock_messages.append({"role": "assistant", "content": assistant_content})

                    # 持久化 AI 回复
                    if st.session_state.get("conv_id"):
                        add_message(
                            st.session_state.conv_id,
                            "assistant",
                            final_text,
                            chart_paths=chart_paths if chart_paths else None,
                        )

                except Exception as e:
                    logger.error(f"调用 Agent 失败: {e}", exc_info=True)
                    error_msg = f"分析出错：{str(e)}"
                    st.error(error_msg)
                    assistant_content = {
                        "text": error_msg,
                        "charts": {},
                        "metrics": {},
                    }
                    st.session_state.stock_messages.append({"role": "assistant", "content": assistant_content})

                    # 持久化错误消息
                    if st.session_state.get("conv_id"):
                        add_message(st.session_state.conv_id, "assistant", error_msg)
