# LegalRAG — Testing Notes & QA Log

## Response Time Measurements

Tested with `Contract.pdf` loaded, queries submitted sequentially.
Measured using `Full query-to-answer` print in terminal.

| Query | Description | Time (s) |
|-------|-------------|----------|
| Q1 | "If a client requests their file from storage 45 days after receiving the letter stating their estate plan is complete, what fee will they be charged?" | 15.34 s |
| Q2 | "What are the exact financial and employee thresholds required for a subconsultant to be bound by the insurance provisions in Section 6?" | 11.32 s |
| Q3 | "What exact step must the Consultant take if there is a conflict between the laws and lawful regulations of different government entities having jurisdiction over the project?" | 13.48 s |
| Q4 | "Within how many days of the Commission's receipt of the Consultant's final billing must the Commission pay allowable costs incurred up to the date of termination for convenience?" | 18.20 s |
| Q5 | "What exact percentage of the last annual policy premium is deemed reasonable for purchasing prior acts or tail coverage under a \"Claims Made\" form?" | 17.68 s |

**Min:** 11.32 s  
**Max:** 18.20 s  
**Average:** 15.20 s  

---

## Test Scenarios

### Upload & Processing

| # | Scenario | Expected | Result | Notes |
|---|----------|----------|--------|-------|
| 1 | Upload valid text-based PDF | Processes successfully, sidebar shows doc card | PASS | |
| 2 | Upload same PDF twice | "Already loaded" info message, no re-embedding | PASS | |
| 3 | Upload password-protected PDF | st.error with clear message | PASS | |
| 4 | Upload scanned PDF (image-only) | st.warning about scanned document | PASS | |
| 5 | Upload multiple PDFs at once | All processed, all appear in sidebar | PASS | |

### Query & Answer

| # | Scenario | Expected | Result | Notes |
|---|----------|----------|--------|-------|
| 6 | Submit blank query | st.warning "Please enter a question" | PASS | |
| 7 | Submit query over 500 chars | st.warning about too long | PASS | |
| 8 | Ask question covered in document | Accurate answer with citation | PASS | |
| 9 | Ask question NOT in document | "No relevant passages found" message | PASS | |
| 10 | Ask follow-up referencing previous answer | Correct context-aware response | PASS | |
| 11 | Submit query before uploading any PDF | st.warning to upload first | PASS | |

### UI & Session

| # | Scenario | Expected | Result | Notes |
|---|----------|----------|--------|-------|
| 12 | Clear Chat History button | Clears chat, keeps documents loaded | PASS | |
| 13 | Clear All Documents button | Resets entire session | PASS | |
| 14 | Source passages expander | Shows chunk text with doc/page/type | PASS | |
| 15 | Debug checkbox in sidebar | Shows last logged JSON entry | PASS | |
| 16 | Welcome screen shows on fresh load | Welcome box visible, no chat input | PASS | |
| 17 | Status indicator updates after upload | Shows green Ready after processing | PASS | |
| 18 | Token count appears after query | Sidebar shows tokens used | PASS / N/A | Gemini may not return count |
| 19 | Query time appears after query | Sidebar shows time in seconds | PASS | |

### Logging

| # | Scenario | Expected | Result | Notes |
|---|----------|----------|--------|-------|
| 20 | Log file created after first query | evaluation/logs/YYYY-MM-DD.json exists | PASS | |
| 21 | Log entry has all required fields | timestamp, session_id, question, answer, contexts, source_metadata | PASS | |
| 22 | Log file grows correctly across queries | Each query appends one entry | PASS | |
| 23 | Logging failure doesn't break chat | Silent pass on logger exception | PASS | |

### Error Handling

| # | Scenario | Expected | Result | Notes |
|---|----------|----------|--------|-------|
| 24 | Network disconnected during query | Clear connectivity error message | PASS | |
| 25 | Rate limit hit | "Too many requests" message | PASS | |
| 26 | Vector store build fails | Rolls back to previous session | PASS | |


## Environment

- **Python:** 3.12  
- **Streamlit:** check with `streamlit 1.58.0`  
- **LangChain:** langchain-classic  
- **Embedding model:** gemini-embedding-2-preview  
- **LLM:** gemini-2.5-flash-lite  
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2  
- **Vector store:** ChromaDB  
- **Test document:** Contract.pdf