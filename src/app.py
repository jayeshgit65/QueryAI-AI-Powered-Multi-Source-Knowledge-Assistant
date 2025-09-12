import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Import functions
from document_processing import get_pdf_documents, get_url_documents, get_text_chunks, create_vectorstore
from rag_pipeline import get_response

load_dotenv()

# Set page title and project name
st.set_page_config(page_title="QueryAI: AI-powered Multi-Source Knowledge Assistant", page_icon="🤖")
st.title("QueryAI💡")
st.subheader("AI-powered Multi-Source Knowledge Assistant")

# Sidebar for sources
with st.sidebar:
    st.header("📚 Data Sources")
    num_sources = st.number_input("How many sources?", min_value=1, max_value=10, value=1)

    sources, source_types = [], []

    for i in range(num_sources):
        st.subheader(f"Source {i+1}")
        source_type = st.selectbox(f"Type", ["URL", "PDF"], key=f"type_{i}")
        if source_type == "URL":
            url = st.text_input("Enter URL", key=f"url_{i}")
            if url:
                sources.append(url)
                source_types.append("URL")
        else:
            pdf = st.file_uploader("Upload PDF", type="pdf", key=f"pdf_{i}")
            if pdf:
                sources.append(pdf)
                source_types.append("PDF")

    if st.button("🔄 Process Sources"):
        if sources:
            with st.spinner("Processing sources..."):
                all_documents = []

                urls = [s for s, t in zip(sources, source_types) if t == "URL"]
                pdfs = [s for s, t in zip(sources, source_types) if t == "PDF"]

                # Process PDFs
                if pdfs:
                    st.info("📄 Processing PDFs...")
                    pdf_docs = get_pdf_documents(pdfs)
                    if pdf_docs:
                        pdf_chunks = get_text_chunks(pdf_docs)
                        all_documents.extend(pdf_chunks)
                        st.success(f"✅ {len(pdfs)} PDFs → {len(pdf_chunks)} chunks")
                    else:
                        st.warning("⚠️ No text found in PDFs")

                # Process URLs
                if urls:
                    st.info("🌐 Processing URLs...")
                    url_docs = get_url_documents(urls)
                    if url_docs:
                        url_chunks = get_text_chunks(url_docs)
                        all_documents.extend(url_chunks)
                        st.success(f"✅ {len(urls)} URLs → {len(url_chunks)} chunks")
                    else:
                        st.warning("⚠️ No content found from URLs")

                # Vectorstore
                if all_documents:
                    st.info("🔗 Creating vector database...")
                    vectorstore = create_vectorstore(all_documents)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success("✅ Vector database created! Start chatting ➡️")
                    else:
                        st.error("❌ Failed to create vector database")
                else:
                    st.error("❌ No content processed!")

# Main Chat UI
if "vectorstore" not in st.session_state:
    st.info("Add and process your sources to start chatting")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm ready to answer questions from your documents.")
        ]

    user_query = st.chat_input("💭 Ask me anything...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.spinner("🤔 Thinking..."):
            response = get_response(user_query, st.session_state.vectorstore, st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=response))

    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
        else:
            with st.chat_message("user"):
                st.write(msg.content)