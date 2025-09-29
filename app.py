# app.py

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from utils import extract_text_from_pdf, get_summary, initialize_llm
from vector_store import create_vector_store_from_text
from qa_chain import get_qa_chain

st.set_page_config(page_title="End to End RAG Chatbot", layout="wide")
st.title("📄 End to End RAG Chatbot")

# Tabs for PDF Chatbot and Direct Chat
tab1, tab2 = st.tabs(["📄 PDF Chatbot", "💬 Chat with AI"])

# -------- Tab 1: PDF Chatbot --------
with tab1:
    st.header("Upload PDF and Chat")

    if "chat_history_pdf" not in st.session_state:
        st.session_state.chat_history_pdf = []

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    summary_mode = st.selectbox("Select Summary Type", ["short", "detailed", "bullet"], key="summary_select")

    if uploaded_file:
        with st.spinner("Reading and processing PDF..."):
            raw_text = extract_text_from_pdf(uploaded_file)

        if raw_text:
            # Display Summary
            st.subheader("📃 Summary")
            summary = get_summary(raw_text, summary_mode)
            st.write(summary)

            # Q&A Chat Interface
            st.subheader("💬 Ask Questions About the PDF")

            vectorstore = create_vector_store_from_text(raw_text)
            qa_chain = get_qa_chain(vectorstore)

            user_query_pdf = st.text_input("Enter your question about the PDF:", key="pdf_chat")

            if user_query_pdf:
                with st.spinner("Generating answer..."):
                    response = qa_chain.run(user_query_pdf)

                # Store in session
                st.session_state.chat_history_pdf.append((user_query_pdf, response))

            if st.session_state.chat_history_pdf:
                for q, a in reversed(st.session_state.chat_history_pdf):
                    st.write(f"**You:** {q}")
                    st.write(f"**Bot:** {a}")

            if st.button("Clear PDF Chat", key="clear_pdf_chat"):
                st.session_state.chat_history_pdf = []

        else:
            st.error("Could not extract text from PDF.")

# -------- Tab 2: Direct Chat with AI --------
with tab2:
    st.header("Talk Directly with AI")

    if "chat_history_direct" not in st.session_state:
        st.session_state.chat_history_direct = []

    user_question = st.text_input("Type your question:", key="direct_chat")

    if st.button("Send", key="send_direct_chat"):
        if user_question.strip() != "":
            with st.spinner("Thinking..."):
                llm = initialize_llm()
                response = llm.invoke(user_question)
                st.session_state.chat_history_direct.append((user_question, response.content))
        else:
            st.warning("Please enter a question before sending.")

    if st.session_state.chat_history_direct:
        for q, a in reversed(st.session_state.chat_history_direct):
            st.write(f"**You:** {q}")
            st.write(f"**Bot:** {a}")

    if st.button("Clear Direct Chat", key="clear_direct_chat"):
        st.session_state.chat_history_direct = []
