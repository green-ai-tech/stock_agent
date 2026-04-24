import streamlit as st
from agents import create_base_agent
from utils import get_logger
from utils.auth import check_login, get_user_by_username

logger = get_logger(__name__)

st.set_page_config(page_title="登录", page_icon="🔐", layout="centered")

if st.session_state.get("logged_in", False):
    st.switch_page("pages/home.py")

# 自定义CSS
st.markdown("""
<style>
    div[data-testid="stForm"] {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.02);
        border: 1px solid #f0f0f0;
    }
    .stTextInput input {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.2);
        outline: none;
    }
    .stButton button {
        border-radius: 40px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton button[kind="primary"] {
        background-color: #3b82f6;
        color: white;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
    .stButton button[kind="secondary"] {
        border: 1px solid #cbd5e1;
        background-color: white;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #f8fafc;
        border-color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>📊 智能股票分析系统</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>登录以使用 AI 股票分析功能</p>", unsafe_allow_html=True)
st.markdown("")

left, center, right = st.columns([1, 2, 1])
with center:
    with st.form("login_form"):
        username = st.text_input("👤 用户名", placeholder="请输入用户名", label_visibility="collapsed")
        password = st.text_input("🔒 密码", type="password", placeholder="请输入密码", label_visibility="collapsed")
        
        col1, col2 = st.columns(2, gap="small")
        with col1:
            submitted = st.form_submit_button("登录", type="primary", width='stretch')
        # 注意：第二个按钮不能在表单内部作为 form_submit_button，所以我们留空 col2
        with col2:
            # 这里放一个空占位，让布局平衡
            st.write("")
        
        if submitted:
            if not username or not password:
                st.error("用户名和密码不能为空")
            elif check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                # 存储 user_id 供会话持久化使用
                user = get_user_by_username(username)
                if user:
                    st.session_state.user_id = user.id
                logger.success(f"登录成功，用户名：{username}")
                st.success("登录成功！正在跳转...")
                agent = create_base_agent()
                st.session_state.agent = agent
                st.switch_page("pages/home.py")
            else:
                st.error("用户名或密码错误")
    
    # 注册按钮放在表单外部，但视觉上紧接表单
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; margin-top: 0.5rem;'>还没有账号？</div>", unsafe_allow_html=True)
        if st.button("📝 注册新账号", width='stretch', type="secondary"):
            st.switch_page("pages/register.py")

st.markdown("---")
st.caption("提示：请联系管理员获取账号", text_alignment="center")