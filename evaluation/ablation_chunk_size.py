import json
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv()
os.environ["RAGAS_MAX_WORKERS"] = "1"

from backend.Document_Loader import doc_loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from backend.rag_pipeline import get_embedding_model, get_llm, get_reranker, RerankRetriever

PDF_PATH        = ROOT_DIR / "data" / "Contract.pdf"
EVAL_DATASET    = ROOT_DIR / "evaluation" / "eval_dataset.json"
RESULTS_PATH    = ROOT_DIR / "evaluation" / "results" / "chunk_size_ablation.json"

CHUNK_CONFIGS = [
    {"chunk_size": 400,  "chunk_overlap": 70},
    {"chunk_size": 800,  "chunk_overlap": 130},
    {"chunk_size": 1200, "chunk_overlap": 200},
]

RESULTS_PATH.parent.mkdir(exist_ok=True)

def build_pipeline_for_chunk_size(chunk_size: int, chunk_overlap: int):
    """
    Rebuilds the full retrieval+rerank+chain pipeline using a fresh
    in-memory Chroma collection for this specific chunk size.
    Retriever weights, reranker, and LLM are untouched — only chunking changes.
    """
    print(f"\n[INFO] Building pipeline for chunk_size={chunk_size}, overlap={chunk_overlap}")

    docs = doc_loader(str(PDF_PATH), "legal")

    for doc in docs:
        doc.metadata["doc_name"]    = PDF_PATH.name
        doc.metadata["pdf_name"]    = PDF_PATH.name
        doc.metadata["doc_type"]    = doc.metadata.get("doc_type", "PDF")
        doc.metadata["page_number"] = doc.metadata.get("page", "N/A")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Produced {len(chunks)} chunks")

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"ablation_{chunk_size}",
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 25}
    )

    llm = get_llm()

    prompt = PromptTemplate(
        template="Generate 3 different questions from this: {question}",
        input_variables=["question"]
    )

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

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and a latest user question which might "
         "reference context in the chat history, formulate a standalone "
         "question. DO NOT answer it, just reformulate if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, rerank_retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional Legal Advisor assistant. "
         "Answer ONLY based on the retrieved context below. "
         "If the answer is not present, say 'I don't know based on the "
         "provided documents.'\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain, chunks

def run_questions_through_pipeline(rag_chain, eval_dataset: list[dict]) -> list[dict]:
    results = []
    total_chunks_retrieved = 0
    total_chars_in_context = 0

    for i, entry in enumerate(eval_dataset):
        question     = entry["question"]
        ground_truth = entry.get("ground_truth", "")

        print(f"  [{i+1}/{len(eval_dataset)}] {question[:60]}...")

        try:
            response = rag_chain.invoke({
                "input": question,
                "chat_history": []
            })
            answer      = response["answer"]
            source_docs = response.get("context", [])

            contexts = [d.page_content for d in source_docs]

            total_chunks_retrieved += len(source_docs)
            total_chars_in_context += sum(len(c) for c in contexts)

            results.append({
                "question":     question,
                "answer":       answer,
                "contexts":     contexts if contexts else [" "],
                "ground_truth": ground_truth,
            })

            time.sleep(1)

        except Exception as e:
            print(f"    [SKIP] Error on question {i+1}: {type(e).__name__}: {e}")
            time.sleep(3)
            continue

    avg_chunks = total_chunks_retrieved / max(len(results), 1)
    avg_chars  = total_chars_in_context / max(total_chunks_retrieved, 1)
    avg_tokens_per_chunk = int(avg_chars / 4)
    avg_total_context_tokens = int(avg_tokens_per_chunk * avg_chunks)

    stats = {
        "avg_chunks_retrieved":     round(avg_chunks, 2),
        "avg_tokens_per_chunk":     avg_tokens_per_chunk,
        "avg_total_context_tokens": avg_total_context_tokens,
    }

    return results, stats

def score_with_ragas(results: list[dict]) -> dict:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from datasets import Dataset

    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    )

    scoreable = [r for r in results if r.get("ground_truth", "").strip()]

    if not scoreable:
        print("[WARN] No entries with ground_truth — skipping context_recall, "
              "scoring with 3 metrics only")
        scoreable = results
        metrics = [faithfulness, answer_relevancy, context_precision]
    else:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    ds = Dataset.from_list([{
        "question":     r["question"],
        "answer":       r["answer"],
        "contexts":     r["contexts"],
        "ground_truth": r.get("ground_truth", ""),
    } for r in scoreable])

    try:
        result = evaluate(ds, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)
        scores = {}
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if m in result:
                scores[m] = round(float(result[m]), 4)
        return scores
    except Exception as e:
        import traceback
        print(f"[ERROR] RAGAS scoring failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {}

def inspect_clause_boundaries(chunks_400: list, sample_size: int = 5):
    print("\n" + "="*60)
    print("MANUAL INSPECTION — 400-char chunk boundaries")
    print("="*60)
    print("Read each chunk's START and END. Does it begin/end mid-sentence")
    print("or mid-clause? Note this in your testing_notes.md.\n")

    step = max(1, len(chunks_400) // sample_size)
    for i, chunk in enumerate(chunks_400[::step][:sample_size]):
        print(f"--- Chunk {i+1} ---")
        print(f"START: ...{chunk.page_content[:80]}")
        print(f"END:   {chunk.page_content[-80:]}...")
        print()

if __name__ == "__main__":
    print("="*60)
    print("Day 22 — Chunk Size Ablation Study")
    print("="*60)

    if not EVAL_DATASET.exists():
        print(f"[ERROR] Eval dataset not found at {EVAL_DATASET}")
        print("Run evaluation/ragas_eval.py first to generate it.")
        sys.exit(1)

    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    print(f"[INFO] Loaded {len(eval_dataset)} questions from fixed eval dataset")

    all_results = {}

    for config in CHUNK_CONFIGS:
        chunk_size    = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]

        print(f"\n{'='*60}")
        print(f"TESTING chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        print(f"{'='*60}")

        rag_chain, chunks = build_pipeline_for_chunk_size(chunk_size, chunk_overlap)

        if chunk_size == 400:
            inspect_clause_boundaries(chunks)

        print(f"[INFO] Running {len(eval_dataset)} eval questions through pipeline.")
        results, stats = run_questions_through_pipeline(rag_chain, eval_dataset)

        print(f"[INFO] Scoring with RAGAS...")
        scores = score_with_ragas(results)

        all_results[str(chunk_size)] = {
            "chunk_size":    chunk_size,
            "chunk_overlap": chunk_overlap,
            "scores":        scores,
            "cost_stats":    stats,
            "questions_evaluated": len(results),
        }

        print(f"\n[DONE] chunk_size={chunk_size} results:")
        print(json.dumps(all_results[str(chunk_size)], indent=2))

    output = {
        "date": time.strftime("%Y-%m-%d"),
        "description": "Ablation study comparing chunk sizes 400, 800, 1200 "
                        "on fixed Day 21 evaluation dataset. Retriever weights, "
                        "reranker, and LLM held constant.",
        "results": all_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Saved comparison to {RESULTS_PATH}")

    print("\n" + "="*80)
    print("CHUNK SIZE ABLATION — FINAL COMPARISON")
    print("="*80)
    print(f"{'Chunk Size':<12}{'Faithfulness':<14}{'Ans.Relevancy':<15}"
          f"{'Ctx.Precision':<15}{'Ctx.Recall':<12}{'Avg Chunks':<12}{'Avg Tokens':<10}")
    print("-"*80)

    for size_str, data in all_results.items():
        s = data["scores"]
        c = data["cost_stats"]
        print(
            f"{data['chunk_size']:<12}"
            f"{s.get('faithfulness', '—'):<14}"
            f"{s.get('answer_relevancy', '—'):<15}"
            f"{s.get('context_precision', '—'):<15}"
            f"{s.get('context_recall', '—'):<12}"
            f"{c.get('avg_chunks_retrieved', '—'):<12}"
            f"{c.get('avg_total_context_tokens', '—'):<10}"
        )

    print("="*80)
    print("\nReview the table above. Update evaluation/testing_notes.md with")
    print("your decision on which chunk size to keep, citing the scores.")