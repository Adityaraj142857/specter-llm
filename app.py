import streamlit as st
import ollama
from ingestion.pdf_reader import extract_text_from_pdf, split_into_chunks
from ingestion.red_flag_detector import summarise_chunk, RED_FLAG_PROMPT
from query.qa import answer_question
from storage.database import (
    init_db, save_document, save_flags,
    save_question, get_all_documents,
    get_flags_for_document, get_questions_for_document
)

# Always init database on startup
init_db()

st.set_page_config(
    page_title="Specter LLM",
    page_icon="⚖️",
    layout="centered"
)

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Analyse Contract", "History"])

# ─────────────────────────────────────────
# PAGE 1 — Analyse a contract
# ─────────────────────────────────────────
if page == "Analyse Contract":
    st.title("⚖️ Specter LLM")
    st.markdown("Upload a legal contract. Get red flags and ask questions in plain English.")
    st.divider()

    uploaded_file = st.file_uploader("Upload a contract PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()

        if "contract_text" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
            with st.spinner("Reading contract..."):
                text = extract_text_from_pdf(pdf_bytes)
                st.session_state["contract_text"] = text
                st.session_state["filename"] = uploaded_file.name
                st.session_state["flags"] = None
                st.session_state["summary"] = ""
                st.session_state["document_id"] = None
            st.success(f"Loaded: {uploaded_file.name} ({len(text.split())} words)")

        contract_text = st.session_state["contract_text"]

        st.divider()
        st.subheader("🚩 Red Flag Analysis")

        if st.button("Detect Red Flags", type="primary"):
            progress = st.progress(0, text="Starting analysis...")
            chunks = split_into_chunks(contract_text)
            limited = chunks[:10]
            all_flags = []
            running_summary = ""

            for i, chunk in enumerate(limited):
                progress.progress(
                    int((i + 1) / len(limited) * 100),
                    text=f"Reading section {i+1} of {len(limited)}..."
                )
                running_summary = summarise_chunk(chunk, running_summary)
                prompt = RED_FLAG_PROMPT.format(summary=running_summary, chunk=chunk)
                response = ollama.chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = response["message"]["content"].strip()
                if "NO FLAGS FOUND" in raw:
                    continue
                for entry in raw.split("---"):
                    entry = entry.strip()
                    if not entry:
                        continue
                    flag = {}
                    for line in entry.split("\n"):
                        if line.startswith("FLAG:"):
                            flag["title"] = line.replace("FLAG:", "").strip()
                        elif line.startswith("CLAUSE:"):
                            flag["clause"] = line.replace("CLAUSE:", "").strip()
                        elif line.startswith("WHY:"):
                            flag["why"] = line.replace("WHY:", "").strip()
                        elif line.startswith("SEVERITY:"):
                            flag["severity"] = line.replace("SEVERITY:", "").strip()
                    if "title" in flag and "why" in flag:
                        all_flags.append(flag)

            # Save to database
            doc_id = save_document(
                filename=uploaded_file.name,
                word_count=len(contract_text.split()),
                summary=running_summary
            )
            save_flags(doc_id, all_flags)
            st.session_state["flags"] = all_flags
            st.session_state["summary"] = running_summary
            st.session_state["document_id"] = doc_id
            progress.empty()

        if st.session_state.get("flags") is not None:
            flags = st.session_state["flags"]
            if not flags:
                st.success("No major red flags detected.")
            else:
                st.warning(f"{len(flags)} red flag(s) found.")
                for flag in flags:
                    severity = flag.get("severity", "Medium")
                    color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "🟡")
                    with st.expander(f"{color} {flag.get('title', 'Unknown')} — {severity}"):
                        if "clause" in flag:
                            st.markdown(f"**Clause:** {flag['clause']}")
                        st.markdown(f"**Why this matters:** {flag['why']}")

        st.divider()
        st.subheader("💬 Ask a Question")
        st.caption("Ask anything about this contract in plain English.")

        question = st.text_input("Your question", placeholder="e.g. Can either party terminate early?")

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Finding answer..."):
                    answer = answer_question(
                        question,
                        contract_text,
                        summary=st.session_state.get("summary", "")
                    )
                st.markdown("**Answer:**")
                st.markdown(answer)
                st.caption("⚠️ Informational only. Not legal advice.")

                # Save question to database
                if st.session_state.get("document_id"):
                    save_question(st.session_state["document_id"], question, answer)

# ─────────────────────────────────────────
# PAGE 2 — History
# ─────────────────────────────────────────
elif page == "History":
    st.title("📁 Document History")
    st.markdown("All contracts you have analysed previously.")
    st.divider()

    docs = get_all_documents()

    if not docs:
        st.info("No documents analysed yet. Upload a contract to get started.")
    else:
        for doc in docs:
            with st.expander(f"📄 {doc['filename']} — {doc['uploaded_at'][:10]}"):
                st.caption(f"{doc['word_count']} words")

                flags = get_flags_for_document(doc["id"])
                if flags:
                    st.markdown(f"**{len(flags)} red flag(s) found:**")
                    for flag in flags:
                        severity = flag.get("severity", "Medium")
                        color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "🟡")
                        st.markdown(f"{color} **{flag['title']}** — {flag['severity']}")
                        st.caption(flag['why'])
                else:
                    st.markdown("No red flags found.")

                questions = get_questions_for_document(doc["id"])
                if questions:
                    st.divider()
                    st.markdown("**Questions asked:**")
                    for q in questions:
                        st.markdown(f"**Q:** {q['question']}")
                        st.markdown(f"**A:** {q['answer']}")
                        st.caption(q['asked_at'][:16])
