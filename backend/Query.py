from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate,PromptTemplate
from langchain_classic.chains import (create_history_aware_retriever,create_retrieval_chain)
from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from Document_Loader import doc_loader
from dotenv import load_dotenv

load_dotenv()
#Defining the Embedding model
embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

#Defining the chat model
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#Loading the vector store
vector_store=Chroma(
    persist_directory=r"C:\Users\Lenovo\OneDrive\Documents\LegalRAG\data\chroma_db",
    embedding_function=embedding,
    collection_name="sample"
)

#BM25 Retriever to retrieve documents directly from the loaded document no need of embedding
docs=doc_loader("data\Contract.pdf","contract")
bm25_retriever=BM25Retriever.from_documents(docs)
bm25_retriever.k=3

#Vector Retriever from the vector store(Chroma)
retriever=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2}
)

#Prompt Template to generate three 3 different questions from the user's query fro better generation of results
prompt=PromptTemplate(
    template="Generate 3 different questions from this : {question}",
    input_variables=['question']
)

#Multi Query Retriever
multi_query_retriever=MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model,
    prompt=prompt,
    include_original=False
)

#Ensembling both the retrievers->vector_retriever and the bm25_retriever
ensemble_retriever=EnsembleRetriever(
    retrievers=[multi_query_retriever,bm25_retriever],
    weights=[0.7,0.3]
)

#Contextualizing the recent query according to the previous chat history.
contextualize_q_system_prompt=(
    """
    Given a chat history and a latest user quetsion
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history.DO NOT answer the question,just
    reformulate it if needed and otherwise return it as it is.
    """
)

#Making a prompt from the contextualized query
contextualize_q_prompt=ChatPromptTemplate.from_messages(
    [("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
    ]
)

#History aware retriever
history_aware_retriever=create_history_aware_retriever(
    model,ensemble_retriever,contextualize_q_prompt
)

#Final prompt which takes the contextualized query and gives the response accordingly
qa_system_prompt=(
    """
    You are a professional Legal Advisor assistant for question-answering the
    questions asked by the user.Use the following pieces of retrieved
    context to answer the question.If you don't know the answer, just
    say I don't know.Use five sentences maximum to answer the question
    in concise.You can use bullet points if required to give clear explanation.
    \n\n
    {context}
    """
)

#Final prompt template which takes the contextualized query as the input and gives the response accordingly
qa_prompt=ChatPromptTemplate.from_messages(
    [("system",qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
    ]
)

#Questions-Answer Chain creates a chain that combines all document text into a single context prompt 
#and passes it to the LLM to answer a question.
question_answer_chain=create_stuff_documents_chain(model,qa_prompt)

#Combines the retriever chain and the QA chain into a final RAG Pipeline
#It fetches the relevant document first the passes them to the LLM to get the response.
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

#Stores the chat history
chat_history=[]

#A BASIC CHATBOT
while(True):
    query=input("\nUser: ")
    if(query.lower()=="exit"):
        break
    ans=rag_chain.invoke({"input":query,"chat_history":chat_history})
    print(f"AI: {ans["answer"]}")
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=ans["answer"]))