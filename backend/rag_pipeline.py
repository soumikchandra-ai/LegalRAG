from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate,PromptTemplate
from langchain_classic.chains import (create_history_aware_retriever,create_retrieval_chain)
from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
import re
from pathlib import Path
from backend.Document_Loader import doc_loader
from dotenv import load_dotenv

load_dotenv()
#Defining the Embedding model
embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

#Defining the chat model
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

##Reranking of retrieved documents
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class RerankRetriever(BaseRetriever):
    base_retriever:any
    top_k:int=3
    
    def _get_relevant_documents(self, query:str)->List[Document]:
        docs=self.base_retriever.invoke(query)
        if not docs:
            return []
        
        section_pattern=re.compile(r'(?:^\s*|\n)(\d+)\.\s+([A-Z\s,]{3,})(?:\.|\n|$)')
        page_pattern = re.compile(r'Page\s+(\d+)', re.IGNORECASE)
        
        pairs=[]
        for doc in docs:
            pairs.append((query,doc.page_content))
            
        scores=reranker.predict(pairs)
        ranked_docs=sorted(zip(docs,scores),key=lambda x:x[1],reverse=True)
        final_docs=[]
        for doc,scores in ranked_docs[:self.top_k]:
            
            metadata_page=doc.metadata.get("page",None)
            page_num=str(metadata_page+1) if metadata_page is not None else "Unknown"
            
            page_matches = page_pattern.findall(doc.page_content)
            if page_matches:
                page_num = page_matches[-1]
                
            detected_section = "Unknown Section"
            section_matches = list(section_pattern.finditer(doc.page_content))
            if section_matches:
                last_match = section_matches[-1]
                detected_section = f"Section {last_match.group(1)} ({last_match.group(2).strip()})"
            else:
                for full_doc in docs:
                    if doc.page_content[:50] in full_doc.page_content:
                        full_text = full_doc.page_content
                        chunk_idx = full_text.find(doc.page_content[:50])
                        all_sections = list(section_pattern.finditer(full_text))
                        for m in reversed(all_sections):
                            if m.start() <= chunk_idx:
                                detected_section = f"Section {m.group(1)} ({m.group(2).strip()})"
                                break
                        break
            print(doc[0].metadata)
            pdf_name=doc.metadata.get("pdf_name","Unknown")
            doc.page_content = (
                f"[Section Context: {detected_section} | Verified Page: {page_num} | PDF Name: {pdf_name}]\n"
                f"{doc.page_content}"
            )
            final_docs.append(doc)
            
        return final_docs

#Stores the chat history
chat_history=[]

def get_rag_chain(vector_store,documents):
        #BM25 Retriever to retrieve documents directly from the loaded document no need of embedding
        bm25_retriever=BM25Retriever.from_documents(documents)
        bm25_retriever.k=10

        #Vector Retriever from the vector store(Chroma)
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":25}
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
        
        rerank_retriever=RerankRetriever(base_retriever=ensemble_retriever,top_k=3)

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
            model,rerank_retriever,contextualize_q_prompt
        )

        #Final prompt which takes the contextualized query and gives the response accordingly
        qa_system_prompt=(
            """
            You are a professional Legal Advisor assistant for question-answering.
            Use the following pieces of retrieved context to answer the question using a maximum of five sentences. 
            You can use bullet points if required to give clear explanations.
            CRITICAL CITATION RULES:
            1. Every response MUST explicitly list the Document Name, Page Number, and Section/Clause number at the absolute beginning of your response.
            2. Read the structural token header string injected at the start of each context block: e.g., '[Section Context: Section 2 (COMPENSATION) | Verified Page: 1 |PDF File: Name of the PDF from which the answer is generated]'. Use these exact values.
            3. If a clause line begins with an embedded alphabet sub-letter marker (e.g., 'H. Invoices shall be...'), merge the parent section and subsection into a standardized legal citation pattern: "Section 2.H".
            4. If the section or page is listed as "Unknown", look closely at the text strings nearby to trace the context, or do not guess. If you do not know the answer, say "I don't know."

            Context:
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

        return rag_chain