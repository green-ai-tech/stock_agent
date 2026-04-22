# pages/main_page.py
import streamlit as st
import time
from datetime import datetime
import PIL.Image as Image
from uis.ui_ai_assitant import ui_ai_assitant
from uis.ui_setting import ui_setting
from uis.ui_stock_analysis import ui_stock_analysis

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

def logout():
    """退出登录"""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.switch_page("pages/login.py")

with st.sidebar:
    # ========== 新增：用户状态与操作入口 ==========
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
    st.divider()  # 可选分割线
    # ========== 原有页面选择 ==========
    with st.container(height=250):
        if st.button("股市数据分析", width="stretch", type="primary" if st.session_state.page == "股市数据分析" else "secondary"):
            st.session_state.page = "股市数据分析"
            st.rerun()
        if st.button("股市智慧助手", width="stretch", type="primary" if st.session_state.page == "股市智慧助手" else "secondary"):
            st.session_state.page = "股市智慧助手"
            st.rerun()
        if st.button("模型参数设置", width="stretch", type="primary" if st.session_state.page == "模型参数设置" else "secondary"):
            st.session_state.page = "模型参数设置"
            st.rerun()
    # ========== 原有 tabs ==========
    with st.container(height=150, horizontal_alignment="center"):
        tab1, tab2, tab3 = st.tabs(["模型参数", "词嵌入模型", "知识库"]) 
        with tab1:
            st.markdown("模型参数")
        with tab2:
            st.markdown("词嵌入")
        with tab3:
            st.markdown("知识库")
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
    ui_ai_assitant()
    
        
elif st.session_state.page == "模型参数设置":
    ui_setting()
    

