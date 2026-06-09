from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from Document_Loader import doc_loader

load_dotenv()

embed=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

vector_store=Chroma(
    persist_directory=r"C:\Users\Lenovo\OneDrive\Documents\LegalRAG\data\chroma_db",
    embedding_function=embed,
    collection_name="sample"
)
retriever=vector_store.as_retriever(
    search_type="similarity",
    kwargs={"k":2}
)

prompt=PromptTemplate(
    template="Generate 3 different questions for this : {question}",
    input_variables=['question']
)

multi_query_retrievr=MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model,
    prompt=prompt,
    include_original=False
)

docs=doc_loader("data\Contract.pdf","contract")
query="Does the agency own the copyright for the things I create?"
retrieved_docs=multi_query_retrievr.invoke(query)

for docs in retrieved_docs:
    print(docs.page_content)