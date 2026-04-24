# pages/main_page.py
import streamlit as st
import time
from datetime import datetime
from uis.ui_ai_assistant import ui_ai_assistant
from uis.ui_setting import ui_setting
from uis.ui_stock_analysis import ui_stock_analysis
from utils.chat_history import (
    create_conversation, get_user_conversations,
    delete_conversation, get_messages, generate_title,
)
from utils.db import User

st.set_page_config(
    page_title="主页",
    page_icon="🏠",
    layout="wide"
)

# 检查登录状态
if not st.session_state.get("logged_in", False):
    st.warning("请先登录")
    time.sleep(1)
    st.switch_page("pages/login.py")

if "page" not in st.session_state:
    st.session_state.page = "股市数据分析"

# 初始化会话相关状态
if "conv_id" not in st.session_state:
    st.session_state.conv_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stock_messages" not in st.session_state:
    st.session_state.stock_messages = []


def _agent_type_for_page(page: str) -> str:
    """根据当前页面返回对应的 agent_type"""
    if page == "股市智慧助手":
        return "stock"
    return "base"


def _load_conversation(conv_id: int):
    """从数据库加载会话消息到 session_state"""
    db_messages = get_messages(conv_id)
    agent_type = _agent_type_for_page(st.session_state.page)

    if agent_type == "stock":
        loaded = []
        for m in db_messages:
            if m.role == "user":
                loaded.append({"role": "user", "content": m.content})
            else:
                import json
                charts = m.chart_paths if m.chart_paths else {}
                loaded.append({"role": "assistant", "content": {"text": m.content, "charts": charts, "metrics": {}}})
        st.session_state.stock_messages = loaded
    else:
        loaded = []
        for m in db_messages:
            loaded.append({"role": m.role, "content": m.content})
        st.session_state.messages = loaded

    st.session_state.conv_id = conv_id


def _new_conversation():
    """创建新会话"""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    agent_type = _agent_type_for_page(st.session_state.page)
    conv = create_conversation(user_id=user_id, agent_type=agent_type)
    if conv:
        st.session_state.conv_id = conv.id
        st.session_state.messages = []
        st.session_state.stock_messages = []


def logout():
    """退出登录"""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_id = None
    st.switch_page("pages/login.py")


with st.sidebar:
    # ========== 用户状态与操作入口 ==========
    if st.session_state.get("logged_in", False):
        st.markdown(f"### 👤 {st.session_state.username}")
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("pages/change_password.py", label="🔑 修改密码")
        with col2:
            if st.session_state.username == "admin":
                st.page_link("pages/admin_user.py", label="👑 用户管理")
    else:
        st.markdown("### 🚪 未登录")
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("pages/login.py", label="登录")
        with col2:
            st.page_link("pages/register.py", label="注册")
    st.divider()

    # ========== 页面切换 ==========
    with st.container(height=250):
        if st.button("股市数据分析", width="stretch", type="primary" if st.session_state.page == "股市数据分析" else "secondary"):
            st.session_state.page = "股市数据分析"
            st.session_state.conv_id = None
            st.session_state.messages = []
            st.session_state.stock_messages = []
            st.rerun()
        if st.button("股市智慧助手", width="stretch", type="primary" if st.session_state.page == "股市智慧助手" else "secondary"):
            st.session_state.page = "股市智慧助手"
            st.session_state.conv_id = None
            st.session_state.messages = []
            st.session_state.stock_messages = []
            st.rerun()
        if st.button("模型参数设置", width="stretch", type="primary" if st.session_state.page == "模型参数设置" else "secondary"):
            st.session_state.page = "模型参数设置"
            st.session_state.conv_id = None
            st.rerun()

    # ========== 历史会话列表（仅在对话页面显示） ==========
    if st.session_state.page in ("股市数据分析", "股市智慧助手"):
        st.divider()
        agent_type = _agent_type_for_page(st.session_state.page)
        user_id = st.session_state.get("user_id")

        if user_id:
            if st.button("➕ 新建对话", width="stretch", type="primary"):
                _new_conversation()
                st.rerun()

            st.markdown("#### 📋 历史对话")
            conversations = get_user_conversations(user_id, agent_type=agent_type)
            if conversations:
                for conv in conversations:
                    col_btn, col_del = st.columns([5, 1])
                    with col_btn:
                        label = f"{'🔹 ' if conv.id == st.session_state.conv_id else ''}{conv.title}"
                        if st.button(label, key=f"conv_{conv.id}", width="stretch"):
                            _load_conversation(conv.id)
                            st.rerun()
                    with col_del:
                        if st.button("🗑", key=f"del_{conv.id}", width="stretch"):
                            delete_conversation(conv.id)
                            if st.session_state.conv_id == conv.id:
                                st.session_state.conv_id = None
                                st.session_state.messages = []
                                st.session_state.stock_messages = []
                            st.rerun()
            else:
                st.caption("暂无历史对话")

    # ========== 退出登录 ==========
    with st.container(height=100, horizontal_alignment="center"):
        if st.session_state.get("logged_in", False):
            if st.button("🚪 退出登录", width="content"):
                logout()
        st.caption(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", text_alignment="center")

# 主内容区
if st.session_state.page == "股市数据分析":
    ui_stock_analysis()

elif st.session_state.page == "股市智慧助手":
    ui_ai_assistant()

elif st.session_state.page == "模型参数设置":
    ui_setting()
