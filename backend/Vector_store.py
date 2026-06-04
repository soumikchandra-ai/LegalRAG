from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Text_Splitter import split_docs
from dotenv import load_dotenv

load_dotenv()

splitted_docs=split_docs("data\Contract.pdf","contract")

vector_store=Chroma.from_documents(
    documents=splitted_docs,
    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview"),
    persist_directory=r'C:\Users\Lenovo\OneDrive\Documents\LegalRAG\data\chroma_db',
    collection_name="sample"
)