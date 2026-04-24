import streamlit as st

pg = st.navigation([
    st.Page("pages/home.py", title="主页", icon="🔥"),
    st.Page("pages/login.py", title="登录", icon=":material/favorite:"),
    st.Page("pages/register.py", title="注册", icon="📝"),                 # 新增
    st.Page("pages/change_password.py", title="修改密码", icon="🔑"),       # 新增
    st.Page("pages/admin_user.py", title="用户管理", icon="👑"),            # 新增（注意你的文件名是admin_user.py）
], position="hidden")

pg.run()