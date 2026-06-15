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

@st.cache_resource
def build_rag_chain(document_key):
    documents=st.session_state["cached_documents"]
    vector_store=create_vector_store(documents)
    rag_chain=get_rag_chain(vector_store,documents)
    return vector_store,rag_chain

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
    
if "cached_documents" not in st.session_state:
    st.session_state.cached_documents=None
    
uploaded_pdfs=st.file_uploader("Upload a Legal PDF",type=["pdf"],accept_multiple_files=True)

if uploaded_pdfs:
    if st.button("Upload and Process PDF"):
        with st.spinner("Processing PDF"):
            all_documents=[]
            
            for uploaded_pdf in uploaded_pdfs:
                with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_path=tmp.name
                    
                docs=doc_loader(tmp_path,"legal")
                for doc in docs:
                    doc.metadata["pdf_name"]=uploaded_pdf.name
                all_documents.extend(docs)
                
                os.unlink(tmp_path)
                
                if uploaded_pdf.name not in st.session_state.uploaded_docs:
                    st.session_state.uploaded_docs.append(uploaded_pdf.name)
                    
            st.session_state.cached_documents=all_documents
            vector_store,rag_chain=build_rag_chain("_".join([pdf.name for pdf in uploaded_pdfs]))
            
            st.session_state.vector_store=vector_store
            st.session_state.rag_chain=rag_chain
            
        st.success("PDF Processed Successfully.")
        
if st.session_state.uploaded_docs:
    st.subheader("Uploaded Documents")
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
        with st.spinner("Thinking...."):
            response=st.session_state.rag_chain.invoke({
            "input":user_query,
            "chat_history":st.session_state.chat_history
        })
        answer=response["answer"]
        
        st.write("Answer")
        st.write(answer)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        st.session_state.chat_history.append(AIMessage(content=answer))
        
if st.session_state.chat_history:
    st.subheader("Chat History")

    for msg in st.session_state.chat_history:

        if isinstance(msg, HumanMessage):
            st.markdown(f"**Query:** {msg.content}")

        elif isinstance(msg, AIMessage):
            st.markdown(f"**AI:** {msg.content}")

        st.divider()