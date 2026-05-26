import streamlit as st
from auth.session import get_current_user

user = get_current_user()
if user is None or user["role"] != "legal":
    st.warning("Access denied. Legal team only.")
    st.stop()

st.title("Legal Dashboard")
st.markdown("This is where escalated questions will appear for review.")
st.divider()
st.info("No escalations yet. When users get low-confidence answers, they will appear here.")
