# pages/login.py
import streamlit as st
import time
from utils import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="登录",
    page_icon="🔐",
    # layout="wide",
    layout="centered"
)

# 如果已经登录，直接跳转到主页面
if st.session_state.get("logged_in", False):
    st.switch_page("pages/home.py")

def check_login(username: str, password: str) -> bool:
    """验证登录"""
    # 示例：硬编码验证（实际应查询数据库）
    valid_users = {
        "admin": "123456",
        "user1": "123456",
    }
    return username in valid_users and valid_users[username] == password

# 页面标题
st.markdown("# 登录智能股票分析系统", text_alignment="center")
st.markdown("请填写以下信息进行登录", text_alignment="center")
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
# 登录表单
with col2:
    with st.form("login_form"):
        username = st.text_input("用户名", placeholder="请输入用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        
        col1, col2, col3, col4, col5 = st.columns([2,4,1,4,2])
        with col2:
            submitted = st.form_submit_button("登录", type="primary", width="stretch")
        with col4:
            st.form_submit_button("重置", type="secondary", width="stretch")
        
        if submitted:
            if not username or not password:
                st.error("用户名和密码不能为空")
            elif check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                logger.success(f"登录成功，用户名：{username}")
                st.success("登录成功！正在跳转...")
                time.sleep(0.5)
                st.switch_page("pages/home.py")
            else:
                st.error("用户名或密码错误")

# 提示信息
st.markdown("---")
st.caption("提示：请联系管理员获取账号", text_alignment="center")