import streamlit as st
from utils.auth import create_user

def register_page():
    st.title("新用户注册")
    
    # 如果已经登录，提示并跳转
    if st.session_state.get('logged_in'):
        st.warning("您已登录，如需注册新账号请先退出。")
        if st.button("返回首页"):
            st.switch_page("app.py")  # 根据你的主入口文件名调整
        return
    
    with st.form("register_form"):
        username = st.text_input("用户名", max_chars=50)
        email = st.text_input("电子邮箱（可选）", max_chars=100)
        password = st.text_input("密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        submitted = st.form_submit_button("注册")
        
        if submitted:
            # 表单验证
            if not username or not password:
                st.error("用户名和密码不能为空")
            elif password != confirm_password:
                st.error("两次输入的密码不一致")
            else:
                success = create_user(username, password, email if email else None)
                if success:
                    st.success("注册成功！请前往登录")
                    # 可选：自动跳转到登录页
                    if st.button("去登录"):
                        st.switch_page("pages/login.py")
                else:
                    st.error("用户名已存在，请更换用户名")

if __name__ == "__main__":
    register_page()