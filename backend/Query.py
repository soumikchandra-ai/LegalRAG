from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_classic.chains import (create_history_aware_retriever,create_retrieval_chain)
from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)
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

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

contextualize_q_system_prompt=(
    """
    Given a chat history and a latest user quetsion
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history.DO NOT answer the question,just
    reformulate it if needed and otherwise return it as it is.
    """
)

contextualize_q_prompt=ChatPromptTemplate.from_messages(
    [("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
    ]
)

history_aware_retriever=create_history_aware_retriever(
    model,retriever,contextualize_q_prompt
)

qa_system_prompt=(
    """
    You are a Legal Advisor assistant for question-answering the
    questions asked by the user.Use the following pieces of retrieved
    context to answer the question.If you don't know the answer, just
    say I don't know.Use five sentences maximum to answer the question
    in concise.
    \n\n
    {context}
    """
)

qa_prompt=ChatPromptTemplate.from_messages(
    [("system",qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(model,qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

chat_history=[]
while(True):
    query=input("\nUser:")
    if(query.lower()=="exit"):
        break
    ans=rag_chain.invoke({"input":query,"chat_history":chat_history})
    print(f"AI: {ans["answer"]}")
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=ans["answer"]))