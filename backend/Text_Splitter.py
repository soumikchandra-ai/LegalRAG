from langchain_text_splitters import RecursiveCharacterTextSplitter
from Document_Loader import doc_loader

def split_docs(file_path:str,doc_type:str):
    doc=doc_loader(file_path,doc_type)
    
    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
    )
    
    splitted_docs=text_splitter.split_documents(doc)
    
    return splitted_docs