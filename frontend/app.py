from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
import streamlit as st
st.header("LEGAL RAG")
from backend.Document_Loader import doc_loader
from backend.Text_Splitter import split_docs
from backend.Vector_Store import create_vector_store

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs=[]
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

upload_pdf=st.file_uploader("Upload PDF",type="pdf")

if upload_pdf:
    if st.button("Upload and Process PDF"):
        with st.spinner("Processing PDF"):
            documents=doc_loader(upload_pdf,"legal")
            chunks=split_docs(documents)
            vector_store=create_vector_store(chunks)
            st.session_state.vector_store=vector_store
        st.success("PDF Processed successfully")
            
user_query=st.text_area("Ask query related to the uploaded pdf:")
