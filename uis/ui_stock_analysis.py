import streamlit as st
from utils.chat_history import add_message, create_conversation, generate_title
from utils.logger import logger


def consume_and_yield(iterator):
    full = ""
    for chunk in iterator:
        full += chunk
        yield chunk
    st.session_state.last_full_response = full


def _ensure_conversation():
    """确保当前有会话，没有则自动创建"""
    if st.session_state.get("conv_id"):
        return
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    conv = create_conversation(user_id=user_id, agent_type="base")
    if conv:
        st.session_state.conv_id = conv.id


def ui_stock_analysis():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_full_response" not in st.session_state:
        st.session_state.last_full_response = ""

    if "agent" not in st.session_state:
        st.error("智能体未初始化，请重新登录")
        return

    st.title(body="股市数据分析", width="stretch", text_alignment="center")
    st.caption(body="使用的是免费本地部署的大模型", text_alignment="center")

    # 显示历史聊天信息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"], text_alignment="right")
            else:
                st.markdown(message["content"], text_alignment="left")

    prompt = st.chat_input("输入你的问题：")

    if prompt:
        # 确保有会话
        _ensure_conversation()

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt, text_alignment="right")

        # 内存 + 持久化
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.get("conv_id"):
            # 第一条消息时自动生成标题
            if len(st.session_state.messages) == 1:
                from utils.chat_history import rename_conversation
                rename_conversation(st.session_state.conv_id, generate_title(prompt))
            add_message(st.session_state.conv_id, "user", prompt)

        # 调用模型
        with st.chat_message("ai"):
            with st.spinner("思考中..."):
                response = st.session_state.agent.stream(prompt)
                st.write_stream(consume_and_yield(response), cursor="|")

        # 保存 AI 回复
        ai_content = st.session_state.last_full_response
        st.session_state.messages.append({"role": "ai", "content": ai_content})
        if st.session_state.get("conv_id"):
            add_message(st.session_state.conv_id, "ai", ai_content)
