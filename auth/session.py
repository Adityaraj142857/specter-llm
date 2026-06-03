import streamlit as st

USERS = {
    "alice": {"name": "Alice", "role": "hr", "password": "hr123"},
    "bob": {"name": "Bob", "role": "sde", "password": "sde123"},
    "carol": {"name": "Carol", "role": "external", "password": "ext123"},
    "legal": {"name": "Legal Team", "role": "legal", "password": "legal123"},
}

def login_form():
    st.title("Specter LLM — Legal Q&A Gateway")
    st.markdown("Login to ask questions about your contracts.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["user"] = USERS[username]
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

def get_current_user():
    return st.session_state.get("user", None)

def logout():
    st.session_state.clear()
    st.rerun()
