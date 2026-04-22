import streamlit as st
from utils.auth import verify_password, update_password, get_user_by_username

def change_password_page():
    st.title("修改密码")
    
    if not st.session_state.get('logged_in'):
        st.warning("请先登录")
        st.stop()
    
    username = st.session_state['username']
    
    with st.form("change_pw_form"):
        old_password = st.text_input("当前密码", type="password")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认新密码", type="password")
        submitted = st.form_submit_button("更新密码")
        
        if submitted:
            # 验证旧密码
            user = get_user_by_username(username)
            if not user or not verify_password(old_password, user.password_hash):
                st.error("当前密码错误")
            elif not new_password:
                st.error("新密码不能为空")
            elif new_password != confirm_password:
                st.error("两次输入的新密码不一致")
            else:
                if update_password(username, new_password):
                    st.success("密码修改成功，请重新登录")
                    # 清除登录状态
                    for key in ['logged_in', 'username']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                else:
                    st.error("修改失败，请稍后重试")

if __name__ == "__main__":
    change_password_page()