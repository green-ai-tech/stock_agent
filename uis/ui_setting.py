import streamlit as st
from utils.setting import settings

def ui_setting():
    st.title("⚙️ 模型参数设置")
    st.caption("修改后立即生效，重启后恢复为 .env 默认值")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("当前配置")
        st.code(f"模型: {settings.llm_model}\n"
                f"温度: {settings.llm_temperature}\n"
                f"最大Token: {settings.llm_max_tokens}\n"
                f"流式输出: {settings.llm_streaming}\n"
                f"服务地址: {settings.llm_base_url}")

    with col2:
        st.subheader("运行时调整")
        new_temp = st.slider("Temperature", 0.0, 2.0,
                              value=st.session_state.get("runtime_temperature", settings.llm_temperature),
                              step=0.1)
        new_max = st.number_input("Max Tokens", min_value=64, max_value=8192,
                                   value=st.session_state.get("runtime_max_tokens", settings.llm_max_tokens),
                                   step=64)
        if st.button("应用", type="primary"):
            st.session_state["runtime_temperature"] = new_temp
            st.session_state["runtime_max_tokens"] = new_max
            st.success(f"已应用: temperature={new_temp}, max_tokens={new_max}")
            st.info("注意：需重新创建 Agent 才能生效（重新登录或刷新页面）")
