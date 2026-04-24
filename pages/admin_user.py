import streamlit as st
from utils.db import get_db_session, User
from utils.auth import update_password

def admin_users_page():
    st.title("👑 管理员 - 用户管理")
    
    # 权限检查：仅 admin 可访问
    if not st.session_state.get('logged_in'):
        st.warning("请先登录")
        st.stop()
    
    if st.session_state.get('username') != "admin":
        st.error("权限不足：只有管理员可以访问此页面")
        st.stop()
    
    session = get_db_session()
    try:
        # 查询所有用户
        users = session.query(User).order_by(User.id).all()
        if not users:
            st.info("暂无用户")
            return
        
        st.subheader("用户列表")
        # 使用表格展示用户信息
        user_data = []
        for u in users:
            user_data.append({
                "ID": u.id,
                "用户名": u.username,
                "邮箱": u.email or "未设置",
                "注册时间": u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else "",
                "最后登录": u.last_login.strftime("%Y-%m-%d %H:%M:%S") if u.last_login else "从未"
            })
        st.dataframe(user_data, width='stretch')
        
        st.divider()
        st.subheader("重置用户密码")
        
        # 选择要重置密码的用户
        username_list = [u.username for u in users]
        selected_user = st.selectbox("选择用户", username_list)
        new_password = st.text_input("新密码", type="password", key="new_pw")
        confirm_new = st.text_input("确认新密码", type="password", key="confirm_pw")
        
        if st.button("确认重置密码"):
            if not new_password:
                st.error("密码不能为空")
            elif new_password != confirm_new:
                st.error("两次输入的密码不一致")
            else:
                if update_password(selected_user, new_password):
                    st.success(f"用户 {selected_user} 的密码已重置")
                else:
                    st.error("重置失败，请检查日志")
    finally:
        session.close()

if __name__ == "__main__":
    admin_users_page()