import streamlit as st

def consume_and_yield(iterator):
    full = ""
    for chunk in iterator:
        full += chunk
        yield chunk
    # 将累积结果保存到 session_state 的某个变量中
    st.session_state.last_full_response = full

def ui_stock_analysis():
    # 保存聊天历史记录
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
    
    # 处理输入：调用大模型进行推理
    if prompt:
        # 输出用户的输入
        with st.chat_message("user"):
            st.markdown(prompt, text_alignment="right")

        # 把输入的消息存放在状态变量中
        st.session_state.messages.append(
            {
                "role": "user", 
                "content": prompt
            }
        )

        # 调用模型回答
        # 输出AI答复
        with st.chat_message("ai"):   # system, user/human, ai/assitant
            with st.spinner("思考中..."):
                # 调用模型
                # response = st.session_state.agent.invoke(prompt)
                # # 直接输出
                # st.markdown(response, text_alignment="left")
                response = st.session_state.agent.stream(prompt)
                st.write_stream(consume_and_yield(response), cursor="|")   # 流式输出
        # 保存到会话
        st.session_state.messages.append(
            {
                "role"    : "ai",
                "content" :  st.session_state.last_full_response
            }
        )