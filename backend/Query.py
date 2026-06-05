from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store=Chroma(
    persist_directory=r"C:\Users\Lenovo\OneDrive\Documents\LegalRAG\data\chroma_db",
    embedding_function=embedding,
    collection_name="sample"
)

retriever=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2}
)


def format_doc(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parallel_chain=RunnableParallel({
        'context':retriever | format_doc,
        'question':RunnablePassthrough()
    })
    
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()

prompt=PromptTemplate(
    template="""
    You are an experienced Legal Advisor
    Answer only from the provided transcript context.
    If the context is insufficient just tell I don't know.
    
    {context}
    Question:{question}
    """,
    input_variables=['context','question']
)

main_chain = parallel_chain | prompt | llm | parser

def query_answer(question:str):
    return main_chain.invoke(question)

print(query_answer("What is Early Termination?"))
    