#  LegalRAG — Legal Document Q&A System

> An intelligent question-answering system for legal documents powered by Retrieval-Augmented Generation (RAG), LangChain, and Streamlit.

---

##  Problem Statement

Legal professionals routinely work with large volumes of dense documents — contracts, NDAs, regulatory filings, and statutory texts. Extracting specific information manually is slow and error-prone. Existing keyword search tools fail because legal language is highly contextual, and general-purpose LLMs hallucinate statutes and clauses that don't exist.

**LegalRAG** solves this by letting users upload legal PDFs and ask natural-language questions — receiving precise answers grounded in the actual document content, with citations to the exact clause, page, and source document.

---

##  Features

-  **Multi-document support** — Upload and query across multiple legal PDFs simultaneously
-  **Hybrid retrieval** — Combines BM25 (keyword) and FAISS (semantic) search via `EnsembleRetriever`
-  **Query transformation** — `MultiQueryRetriever` generates paraphrases to improve recall on vague legal questions
-  **Cross-encoder reranking** — Reranks top-20 retrieved chunks to the most precise top-5 using `ms-marco-MiniLM-L-6-v2`
-  **Conversational memory** — `ConversationBufferWindowMemory` supports follow-up questions with full context
-  **Source citations** — Every answer cites the document name, page number, and clause it was derived from
-  **Document type tagging** — Tag documents as contract, NDA, terms of service, statute, or other
-  **Hallucination prevention** — LLM is strictly instructed to answer only from retrieved context

---

##  Architecture

```
User Question
      │
      ▼
MultiQueryRetriever (generates 4 paraphrases)
      │
      ▼
EnsembleRetriever (BM25 + FAISS vector search)
      │
      ▼
Cross-Encoder Reranker (top 20 → top 5 chunks)
      │
      ▼
ContextualCompressionRetriever
      │
      ▼
ConversationalRetrievalChain (with memory)
      │
      ▼
GoogleGenerativeAI (gemini-2.5-flash-lite) + Citation-aware Prompt
      │
      ▼
Answer + Structured Citations [Doc, Page, Clause]
```

---

##  Tech Stack

### Frontend
| Tool | Purpose |
|---|---|
| Streamlit | Chat UI, PDF upload, citation display |
| st.chat_message | Native chat interface components |
| st.session_state | Persistent state across reruns |

### RAG & LangChain
| Tool | Purpose |
|---|---|
| langchain | Core chains — ConversationalRetrievalChain, PromptTemplate |
| langchain-community | BM25Retriever, EnsembleRetriever, document loaders |
| langchain-openai | ChatOpenAI LLM and OpenAI embeddings |
| MultiQueryRetriever | Query paraphrasing for better recall |
| EnsembleRetriever | Hybrid BM25 + vector search |
| ContextualCompressionRetriever | Reranker integration |

### ML & Embeddings
| Tool | Purpose |
|---|---|
| text-embedding-3-small | OpenAI embedding model for chunk vectorisation |
| sentence-transformers | Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) |
| PyMuPDF (fitz) | PDF text and metadata extraction |
| pdfplumber | Fallback parser for table-heavy documents |

### Storage
| Tool | Purpose |
|---|---|
| Chroma | Local vector index for semantic search |
| Python list (session) | Conversation history per session |

### Environment
| Tool | Purpose |
|---|---|
| Python 3.11 | Primary runtime |
| python-dotenv | API key management via .env |
| RecursiveCharacterTextSplitter | Chunk size 800, overlap 100 |

---

##  Project Structure

```
LegalRAG/
├── backend/
│   ├── __init__.py
│   ├── Document_Loader.py       # PDF extraction with metadata
│   ├── Text_Splitter.py         # Recursive chunking
│   ├── Vector_Store.py          # FAISS index creation & merging
│   ├── rag_pipeline             # RAG Pipeline which responds to the query
├── frontend/
│   └── app.py                   # Streamlit app entry point
├── data/
│   └── sample_docs/             # Sample legal PDFs for testing
│   └── chroma_db/               # Stores the embedding of the documents uploaded
├── .env                         # API keys (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/soumikchandra-ai/LegalRAG.git
cd LegalRAG
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key
```

### 5. Run the app
```bash
streamlit run frontend/app.py
```

---

##  How to Use

1. Open the app at `http://localhost:8501`
2. Select a document type from the sidebar dropdown (contract, NDA, etc.)
3. Upload one or more legal PDFs using the file uploader
4. Click **Upload & Process** and wait for the success confirmation
5. Type your question in the chat input at the bottom
6. Read the answer and expand the citation cards to verify sources
7. Ask follow-up questions — the system remembers the conversation context

---

##  Requirements

```
langchain
fastapi
uvicorn
langchain-community
langchain-google-genai
python-dotenv
pydantic
pymupdf
faiss-cpu
pypdf
chroma
transformers
sentence-transformers
streamlit
```

---

##  Key Design Decisions

**Why hybrid retrieval?**
Legal documents reference exact clause numbers and defined terms. Pure semantic search misses these. BM25 catches exact keyword matches while vector search handles meaning-based queries. The combination covers both.

**Why a cross-encoder reranker?**
The bi-encoder used for retrieval is fast but approximate. The cross-encoder reads the question and each chunk together, giving a much more accurate relevance score. We retrieve 20 chunks and rerank to the top 5 before sending to the LLM.

**Why strict citation prompting?**
In legal contexts a confident wrong answer is worse than no answer. The system prompt explicitly forbids the LLM from using its own knowledge and requires inline citations for every claim.

---


## 👤 Author

**Soumik Chandra**