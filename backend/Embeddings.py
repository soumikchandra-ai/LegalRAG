from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Text_Splitter import split_docs
from dotenv import load_dotenv

load_dotenv()

def text_embedder(file_path:str,doc_type:str):
    splitted_doc=split_docs("data\Contract.pdf","contract")
    embedding=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768
    )

    content=[]
    for i in range(len(splitted_doc)):
        content.append(splitted_doc[i].page_content)

    vector=embedding.embed_documents(content)
    
    return vector

ans=text_embedder("data\Contract.pdf","contract")
print(ans[0])