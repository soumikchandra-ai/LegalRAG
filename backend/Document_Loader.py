import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

def doc_loader(file_path:str,doc_type:str):
    path=Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError("File does not exists")
    
    if path.suffix.lower()!=".pdf":
        raise ValueError("Upload PDF files")
    
    loader=PyMuPDFLoader(
        file_path=file_path
    )
    document=[]
    docs=loader.load()
    
    for doc in docs:
        if not doc.page_content.strip():
            continue
        
        document.append({
            "page_content":doc.page_content,
            "metadata":{
                **doc.metadata,
                "doc_type":doc_type
            }
        })
        
    return document

ans=doc_loader("data\Contract.pdf","contract")
print(ans[0]["metadata"])