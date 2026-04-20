import streamlit as st

pg = st.navigation([
    st.Page("pages/home.py", title="主页", icon="🔥"),
    st.Page("pages/login.py", title="登录", icon=":material/favorite:"),
], position="hidden")
pg.run()
