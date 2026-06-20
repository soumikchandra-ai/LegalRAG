from pathlib import Path
import sys
ROOT_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT_DIR))
import tempfile,os
import uuid
import streamlit as st
from backend.Document_Loader import doc_loader
from backend.Text_Splitter import split_docs
from backend.Vector_Store import create_vector_store
from backend.rag_pipeline import get_rag_chain
from backend.logger import log_interaction
from langchain_core.messages import HumanMessage,AIMessage

try:
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
        
    if "source_chunks" not in st.session_state:
        st.session_state.source_chunks=[]
        
    if "session_id" not in st.session_state:
        st.session_state.session_id=str(uuid.uuid4())
        
    if "last_log_entry" not in st.session_state:
        st.session_state.last_log_entry=None
        
    with st.sidebar:
        st.header("Session Controls")
        msg_count=len(st.session_state.chat_history)
        st.caption(f"{msg_count} message{'s' if msg_count!=1 else ''} in current session")
        st.caption(f"Session: {st.session_state.session_id[:8]}")
        st.divider()
        
        if st.button("Clear Chat History",use_container_width=True):
            st.session_state.chat_history=[]
            st.session_state.source_chunks=[]
            st.session_state.last_log_entry=None
            
            if st.session_state.rag_chain is not None:
                try:
                    st.session_state.rag_chain.memory.clear()
                except AttributeError:
                    pass
            st.success("Chat History ")
            st.rerun()
        st.divider()
        
        if st.button("Clear all Documents",use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.source_chunks = []
            st.session_state.uploaded_docs = []
            st.session_state.vector_store = None
            st.session_state.rag_chain = None
            st.session_state.cached_documents = None
            st.session_state.last_log_entry=None
            st.success("Everything cleared.")
            st.rerun()
            
        if st.session_state.uploaded_docs:
            st.divider()
            st.subheader("Uploaded Documents")
            for doc in st.session_state.uploaded_docs:
                st.write(f"{doc}")
        st.divider()
        
        show_debug=st.checkbox("Show debug info")
        if show_debug:
            if st.session_state.last_log_entry:
                st.caption("Last Logged Entry:")
                st.json(st.session_state.last_log_entry)
            else:
                st.info("No entries logged")
            
    uploaded_pdfs=st.file_uploader("Upload a Legal PDF",type=["pdf"],accept_multiple_files=True)

    if uploaded_pdfs:
        if st.button("Upload and Process PDF"):
            with st.spinner("Processing PDF"):
                all_documents=[]
                uploaded_failed=False
                
                for uploaded_pdf in uploaded_pdfs:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
                            tmp.write(uploaded_pdf.read())
                            tmp_path=tmp.name
                        docs=doc_loader(tmp_path,"legal")
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        os.unlink(tmp_path) if os.path.exists(tmp_path) else None
                        st.error(f"Could not read **{uploaded_pdf}** , it may be password-protected or corrupted.Deatil: {type(e).__name__}")
                        uploaded_failed=True
                        continue
                    
                    total_chars=sum(len(d.page_content) for d in docs)
                    if total_chars<100:
                        st.warning(f"**{uploaded_pdf.name}** seems to be a scanned document only text-based PDFs are supported.")
                        uploaded_failed=True
                        continue
                    
                    for doc in docs:
                        doc.metadata["doc_name"]=uploaded_pdf.name
                        doc.metadata["pdf_name"]=uploaded_pdf.name
                        doc.metadata["doc_type"]=doc.metadata.get("doc_type","PDF")
                        doc.metadata["page_number"]=doc.metadata.get("page",doc.metadata.get("page_number","N/A"))
                        
                    all_documents.extend(docs)
                    if uploaded_pdf.name not in st.session_state.uploaded_docs:
                        st.session_state.uploaded_docs.append(uploaded_pdf.name)
                            
                    if all_documents:
                        old_vector_store=st.session_state.vector_store
                        old_rag_chain=st.session_state.rag_chain
                        
                        try:
                            st.session_state.cached_documents=all_documents
                            if st.session_state.rag_chain is not None:
                                try:
                                    st.session_state.rag_chain.memory.clear()
                                except AttributeError:
                                    pass
                            
                            vector_store,rag_chain=build_rag_chain("_".join([pdf.name for pdf in uploaded_pdfs]))
                            st.session_state.vector_store=vector_store
                            st.session_state.rag_chain=rag_chain
                            
                        except Exception as e:
                            st.session_state.vector_store=old_vector_store
                            st.session_state.rag_chain=old_rag_chain
                            st.error(f"Can't build document index.Your previous session is still alive.Detail: {type(e).__name__}:{e}")
                            uploaded_failed=True
                            
                    if not uploaded_failed:
                        st.success("PDF Processed Successfully.")
                                  
    st.divider()
    if st.session_state.chat_history:
        st.subheader("Conversation")
        ai_turn_index = 0

        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)

            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

                    if ai_turn_index < len(st.session_state.source_chunks):
                        chunks = st.session_state.source_chunks[ai_turn_index]
                        if chunks:
                            with st.expander("View source passages"):
                                for j, chunk in enumerate(chunks, 1):
                                    meta = chunk.metadata
                                    doc_name    = meta.get("doc_name") or meta.get("pdf_name", "Unknown")
                                    page_number = meta.get("page_number") or meta.get("page", "N/A")
                                    doc_type    = meta.get("doc_type", "PDF")

                                    st.markdown(
                                        f"**Passage {j}** · `{doc_name}` · "
                                        f"Page `{page_number}` · Type `{doc_type}`"
                                    )
                                    st.caption(chunk.page_content[:600])
                                    if j < len(chunks):
                                        st.divider()
                        else:
                            with st.expander("View source passages"):
                                st.info("No source passages were retrieved for this answer.")

                    ai_turn_index += 1

    st.divider()

    st.subheader("Ask Questions")
    user_query = st.text_area(
        "Ask anything related to the uploaded PDF",
        key="query_input",
        placeholder="e.g. What are the termination clauses in this contract?"
    )

    if st.button("Submit Query", type="primary"):
        # Empty query guard
        cleaned_query=user_query.strip()
        if not cleaned_query:
            st.warning("Please enter a question.")
        
        elif(len(cleaned_query)>500):
            st.warning("Your question is too long. Please try to enter a shorter query")

        elif st.session_state.vector_store is None:
            st.warning("Please upload and process a PDF first.")

        else:
            try:
                with st.spinner("Thinking…"):
                    response = st.session_state.rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history,
                    })

                answer     = response["answer"]
                source_docs = response.get("context", [])

                if not source_docs:
                    answer=("No relevant passages found in the uploaded documents for this question. Please try rephrashing.")
                    
                try:
                    entry=log_interaction(question=user_query,
                                    answer=answer,
                                    source_documents=source_docs,
                                    session_id=st.session_state.session_id)
                    st.session_state.last_log_entry=entry
                except Exception:
                    pass
                
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.session_state.source_chunks.append(source_docs)

                st.rerun()
                
            except Exception as e:
                err=str(e).lower()
                if "rate" in err and "limit" in err:
                    st.error("Too many requests please wait a moment and try again.")
                    
                elif any(w in err for w in ["connection","timeout","network","unreachable"]):
                    st.error("Could not reach the AI service.Check your internet connectivity and try again.")
                    
                else:
                    st.error(f"Something went wrong while generating the answer.Try again later.Detail: {type(e).__name__}:{e}")
                    
except Exception as e:
    st.error(f"An unexpected error occured.Please refresh the page and try again.Detail: {type(e).__name__}:{e}")