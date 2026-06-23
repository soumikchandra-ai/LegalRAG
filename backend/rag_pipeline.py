from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
import re
import streamlit as st
import time
from backend.Document_Loader import doc_loader
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class RerankRetriever(BaseRetriever):
    base_retriever: any
    top_k: int = 3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        t0=time.time()
        raw = self.base_retriever.invoke(query)
        print(f"[PROFILE] Ensemble retrieval: {time.time() - t0:.2f}s")
        
        docs:List[Document]=[]
        def extract_docs(obj):
            if isinstance(obj,Document):
                docs.append(obj)
            elif isinstance(obj,List):
                for item in obj:
                    extract_docs(item)
        
        extract_docs(raw)

        if not docs:
            return []

        seen: set = set()
        unique_docs: List[Document] = []
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        docs = unique_docs

        section_pattern = re.compile(r'(?:^\s*|\n)(\d+)\.\s+([A-Z\s,]{3,})(?:\.|\n|$)')
        page_pattern    = re.compile(r'Page\s+(\d+)', re.IGNORECASE)

        t1=time.time()
        reranker=get_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        raw_scores = reranker.predict(pairs)
        print(f"[PROFILE] Reranking {len(docs)} docs: {time.time() - t1:.2f}s")

        ranked_docs = sorted(zip(docs, raw_scores), key=lambda x: x[1], reverse=True)
        top_chunks=[doc for doc,_ in ranked_docs[:self.top_k]]
        avg_chars=sum(len(d.page_content) for d in top_chunks)/max(len(top_chunks),1)
        avg_tokens=int(avg_chars/4)
        total_context_tokens=avg_tokens*self.top_k
        print(f"[PROFILE] Avg chunk ~{avg_tokens} tokens | Total context ~{total_context_tokens} tokens")
        
        final_docs: List[Document] = []
        for doc, score in ranked_docs[:self.top_k]:
            metadata_page = doc.metadata.get("page", None)
            page_num = str(metadata_page + 1) if metadata_page is not None else "Unknown"

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
                    snippet = doc.page_content[:50]
                    if snippet in full_doc.page_content:
                        full_text = full_doc.page_content
                        chunk_idx = full_text.find(snippet)
                        all_sections = list(section_pattern.finditer(full_text))
                        for m in reversed(all_sections):
                            if m.start() <= chunk_idx:
                                detected_section = f"Section {m.group(1)} ({m.group(2).strip()})"
                                break
                        break

            pdf_name = doc.metadata.get("pdf_name", "Unknown")
            doc.page_content = (
                f"[Section Context: {detected_section} | Verified Page: {page_num} | PDF Name: {pdf_name}]\n"
                f"{doc.page_content}"
            )
            final_docs.append(doc)

        return final_docs

def get_rag_chain(vector_store, documents):
    t0=time.time()
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    print(f"[PROFILE] BM25 index build: {time.time() - t0:.2f}s")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 25}
    )

    prompt = PromptTemplate(
        template="Generate 3 different questions from this : {question}",
        input_variables=["question"]
    )
    
    llm=get_llm()

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=prompt,
        include_original=False
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[multi_query_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    rerank_retriever = RerankRetriever(base_retriever=ensemble_retriever, top_k=3)

    contextualize_q_system_prompt = (
        "Given a chat history and a latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. DO NOT answer the question, just "
        "reformulate it if needed and otherwise return it as it is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, rerank_retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are a professional Legal Advisor assistant for question-answering. "
        "Answer ONLY based on the retrieved context below. "
        "If the answer is not present in the context, respond with: "
        "'I don't know based on the provided documents.' "
        "Do not speculate or use outside knowledge.\n\n"
        "Use a maximum of five sentences. You may use bullet points for clarity.\n\n"
        "CRITICAL CITATION RULES:\n"
        "1. Every response MUST explicitly list the Document Name, Page Number, "
        "and Section/Clause number at the absolute beginning of your response.\n"
        "2. Read the structural token header injected at the start of each context block: "
        "e.g., '[Section Context: Section 2 (COMPENSATION) | Verified Page: 1 | PDF Name: contract.pdf]'. "
        "Use these exact values.\n"
        "3. If a clause line begins with an embedded alphabet sub-letter marker "
        "(e.g., 'H. Invoices shall be...'), merge parent section and subsection: 'Section 2.H'.\n"
        "4. If section or page is 'Unknown', do not guess. Say 'I don't know.'\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain