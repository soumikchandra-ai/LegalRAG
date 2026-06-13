from pathlib import Path
import sys
ROOT_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT_DIR))
import tempfile,os
import streamlit as st
from backend.Document_Loader import doc_loader
from backend.Text_Splitter import split_docs
from backend.Vector_Store import create_vector_store
from backend.rag_pipeline import get_rag_chain
from langchain_core.messages import HumanMessage,AIMessage


st.set_page_config(page_title="Legal-RAG",layout="wide")
st.title("Legal RAG Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
    
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs=[]
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None
    
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain=None
    
uploaded_pdf=st.file_uploader("Upload a Legal PDF",type=["pdf"])

if uploaded_pdf is not None:
    st.info(f"Selected File: {uploaded_pdf.name}")
    
    if st.button("Upload and Process PDF"):
        with st.spinner("Processing PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            
            documents = doc_loader(tmp_path, "legal")
            os.unlink(tmp_path) 
            vector_store=create_vector_store(documents)
            rag_chain=get_rag_chain(vector_store,documents)
            st.session_state.rag_chain=rag_chain
            st.session_state.vector_store=vector_store
            st.session_state.uploaded_docs.append(uploaded_pdf.name)
            
        st.success("PDF Processed Successfully.")
        
if st.session_state.uploaded_docs:
    st.subheader("Uploaded DOcuments")
    for doc in st.session_state.uploaded_docs:
        st.write(f"{doc}")
        
st.subheader("Ask Questions")

user_query=st.text_area("Ask anything related to the uploaded PDF")

if st.button("Submit Query"):
    if st.session_state.vector_store is None:
        st.warning("Please upload and Process a PDF first")
    elif not user_query.strip():
        st.warning("Please ask a question.")
    else:
        response=st.session_state.rag_chain.invoke({
            "input":user_query,
            "chat_history":st.session_state.chat_history
        })
        answer=response["answer"]
        
        st.write("###Answer")
        st.write(answer)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        st.session_state.chat_history.append(AIMessage(content=answer))
        
if st.session_state.chat_history:
    st.subheader("Chat History")

    for msg in st.session_state.chat_history:

        if isinstance(msg, HumanMessage):
            st.markdown(f"**Q:** {msg.content}")

        elif isinstance(msg, AIMessage):
            st.markdown(f"**A:** {msg.content}")

        st.divider()