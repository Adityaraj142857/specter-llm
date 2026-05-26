import streamlit as st
from auth.session import get_current_user
from auth.rbac import get_role_label
from query.pipeline import run_query
from models.query import QueryRequest

user = get_current_user()
if user is None:
    st.warning("Please login first.")
    st.stop()

st.title("Ask a Question")
st.markdown(f"You are logged in as **{get_role_label(user['role'])}**. You will only see clauses relevant to your role.")
st.divider()

question = st.text_area("Your question", placeholder="e.g. Can the company assign this contract to someone else?")

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching contracts and generating answer..."):
            req = QueryRequest(
                user_id=user["name"],
                role=user["role"],
                question=question
            )
            result = run_query(req)

        st.divider()

        if result.escalated:
            st.error("⚠️ This question has been escalated to the Legal Team.")
            st.markdown("The system was not confident enough to answer this. Legal will review it.")
        else:
            st.success("✅ Answer")
            st.markdown(result.answer)
            st.caption(f"Confidence: {round(result.confidence * 100)}%")

        st.divider()
        st.markdown("**Source Clauses**")
        for i, clause in enumerate(result.source_clauses):
            with st.expander(f"Clause {i+1} — {clause.clause_type[:60]}"):
                st.markdown(clause.text)
                st.caption(f"Document: {clause.document_id} | Roles: {', '.join(clause.roles)}")

        st.divider()
        st.caption("⚠️ This tool provides informational answers only. It is not legal advice. When in doubt, escalate to your Legal team.")
