from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(documents):
    
    #Splitted the document using Recursive Character Text Splitter
    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
    )
    
    chunks=splitted_docs=text_splitter.split_documents(documents)
    
    return chunks