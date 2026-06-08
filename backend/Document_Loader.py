from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

def doc_loader(file_path:str,doc_type:str):
    #Path of the file to to be analysed
    path=Path(file_path)
    
    #If file path does not exists raise error
    if not path.exists():
        raise FileNotFoundError("File does not exists")
    
    #Check for the file type,i.e., is it a PDF or not
    if path.suffix.lower()!=".pdf":
        raise ValueError("Upload PDF files")
    
    #Load the file from the specified file path
    loader=PyMuPDFLoader(
        file_path=file_path
    )
    #An empty list to store the document
    document=[]
    docs=loader.load()
    
    for doc in docs:
        #If there is no page content in the particular page then just go the next page
        if not doc.page_content.strip():
            continue
        
        #Adding the doc_type of the document in the metadata
        doc.metadata["doc_type"]=doc_type
        document.append(doc)
        
    return document