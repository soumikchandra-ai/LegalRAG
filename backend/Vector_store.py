from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Text_Splitter import split_docs
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def create_vector_store():
    #Splitted the loaded document
    splitted_docs=split_docs("data\Contract.pdf","contract")
    
    #Specifying the embedding model to be used
    embeddings=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        max_retries=6
    )
    
    #Storing the embeddings in the vector database, here it is Chroma
    vector_store=Chroma.from_documents(
        documents=splitted_docs,
        embedding=embeddings,
        persist_directory=r"C:\Users\Lenovo\OneDrive\Documents\LegalRAG\data\chroma_db",
        collection_name="sample"
    )
    
if __name__=="__main__":
    create_vector_store()