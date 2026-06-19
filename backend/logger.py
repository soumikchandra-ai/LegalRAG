import os
import json
from datetime import datetime
from pathlib import Path
from langchain_core.documents import Document

LOG_DIR=Path(__file__).resolve().parent.parent/"evaluation"/"logs"

LOG_DIR.mkdir(parents=True,exist_ok=True)

MAX_LOG_SIZE_BYTES=5*1024*1024 #5MB maximum size of the log file

def _get_log_path()->Path:
    today=datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR/f"{today}.json"

def _check_file_size(log_path:Path):
    if log_path.exists():
        size=log_path.stat().st_size
        if size>MAX_LOG_SIZE_BYTES:
            print(f"[LOGGER WARNING] Log File {log_path.name} is {size/(1024*1024):.1f} MB -needs to archived")

def _extract_context_doc(doc)->str:
    if isinstance(doc,Document):
        return doc.page_content
    return str(doc)

def _extract_metadata(doc)->dict:
    if isinstance(doc,Document):
        meta=doc.metadata
        return {
            "doc_name":meta.get("doc_name") or meta.get("pdf_name","Unknown"),
            "page_number":meta.get("page_number") or meta.get("page","N/A"),
            "doc_type":meta.get("doc_type","PDF")
        }
        
    return {"doc_name":"Unknown","page_number":"N/A","doc_type":"Unknown"}

def log_interaction(question:str,answer:str,source_documents:list,session_id:str)->dict:
    entry={
        "timestamp":datetime.now().isoformat(),
        "session_id":session_id,
        "question":question,
        "answer":answer,
        "contexts":[_extract_context_doc(doc) for doc in source_documents],
        "source_metadata":[_extract_metadata(doc) for doc in source_documents]
    }
    
    log_path=_get_log_path()
    _check_file_size(log_path)
    
    if log_path.exists():
        with open(log_path,"r",encoding="utf-8") as f:
            try:
                existing=json.load(f)
            except json.JSONDecodeError:
                print(f"[LOGGER WARNING] {log_path.name} was malformed, starting fresh.")
                existing=[]
                
    else:
        existing=[]
    
    existing.append(entry)
    
    with open(log_path,"w",encoding="utf-8") as f:
        json.dump(existing,f,indent=2,ensure_ascii=False)
        
    return entry

