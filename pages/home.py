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

# 侧边栏
with st.sidebar:
    with st.container():
        st.markdown(f"### 👤 {st.session_state.username}")
    
    with st.container(height=250):
        # 页面选择
        if st.button("股市数据分析", width="stretch", type="primary" if st.session_state.page == "股市数据分析" else "secondary"):
            st.session_state.page = "股市数据分析"
            st.rerun()
        if st.button("股市智慧助手", width="stretch", type="primary" if st.session_state.page == "股市智慧助手" else "secondary"):
            st.session_state.page = "股市智慧助手"
            st.rerun()
        if st.button("模型参数设置", width="stretch", type="primary" if st.session_state.page == "模型参数设置" else "secondary"):
            st.session_state.page = "模型参数设置"
            st.rerun()
    with st.container(height=150, horizontal_alignment="center"):
        tab1, tab2, tab3 = st.tabs(["模型参数", "词嵌入模型", "知识库"]) 
        with tab1:
            st.markdown("模型参数")
            
        with tab2:
            st.markdown("词嵌入")
            
        with tab3:
            st.markdown("知识库")
            
    
    with st.container(height=100, horizontal_alignment="center"):
        if st.button("🚪 退出登录", width="content",):
            logout()
        st.caption(f"登录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", text_alignment="center")

# 主内容区

if st.session_state.page == "股市数据分析":
    ui_stock_analysis()
    
    
    
elif st.session_state.page == "股市智慧助手":
    ui_ai_assitant()
    
        
elif st.session_state.page == "模型参数设置":
    ui_setting()
    

