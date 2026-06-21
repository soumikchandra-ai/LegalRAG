from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.Text_Splitter import split_docs
from langchain_chroma import Chroma
from dotenv import load_dotenv
from backend.rag_pipeline import get_embedding_model
import streamlit as st
import time

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"


def create_vector_store(documents):
    t0=time.time()
    chunks=split_docs(documents)
    print(f"[PROFILE] Text splitting: {time.time() - t0:.2f}s | {len(chunks)} chunks")
    
    t1=time.time()
    embeddings=get_embedding_model()
    
    #Storing the embeddings in the vector database, here it is Chroma
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="legal_docs"
    )
    print(f"[PROFILE] Embedding + Chroma build: {time.time() - t1:.2f}s")
    
    return vector_store
