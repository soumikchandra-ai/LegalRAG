from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.Text_Splitter import split_docs
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"


def create_vector_store(documents):
    chunks=split_docs(documents)
    #Specifying the embedding model to be used
    embeddings=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        max_retries=6
    )
    
    #Storing the embeddings in the vector database, here it is Chroma
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="legal_docs"
    )
    
    return vector_store
