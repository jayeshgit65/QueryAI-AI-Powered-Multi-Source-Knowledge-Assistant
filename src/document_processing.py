import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# PDF Processing (Extract text from PDFs and return as list of Document objects)
def get_pdf_documents(pdf_files):
    documents = []
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(page_content=text, metadata={"source": pdf.name, "page": i}))
    return documents

# URL Processing (Extract documents from URLs)
def get_url_documents(urls):
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = url
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading {url}: {str(e)}")
    return documents

# Chunking (Split documents into smaller chunks)
def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(documents)

# Embeddings (Load embeddings model (cached to avoid reload))
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# Vectorstore (Create FAISS vectorstore from Document chunks)
def create_vectorstore(documents):
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(documents, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None
