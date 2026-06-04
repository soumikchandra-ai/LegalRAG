from langchain_text_splitters import RecursiveCharacterTextSplitter
from Document_Loader import doc_loader

doc=doc_loader("data\Contract.pdf","Contarct")

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

splitted_doc=text_splitter.split_documents(doc)

print(splitted_doc[0].page_content)