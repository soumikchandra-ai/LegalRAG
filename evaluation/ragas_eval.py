import json
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv()

from backend.Document_Loader import doc_loader
from backend.Text_Splitter import split_docs
from langchain_google_genai import ChatGoogleGenerativeAI

LOG_DIR = ROOT_DIR / "evaluation" / "logs"
EVAL_DATASET = ROOT_DIR / "evaluation" / "eval_dataset.json"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
PDF_PATH = ROOT_DIR / "data" / "Contract.pdf"
SYNTHETIC_COUNT = 20
MIN_REAL_ENTRIES = 5

RESULTS_DIR.mkdir(exist_ok=True)

def load_real_logs() -> list[dict]:
    all_entries = []

    log_files = list(LOG_DIR.glob("*.json"))
    if not log_files:
        print("[WARN] No log files found in evaluation/logs/")
        return []

    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Could not parse {log_file.name} — skipping")
                continue

        for entry in entries:
            question = entry.get("question", "").strip()
            answer = entry.get("answer", "").strip()
            contexts = entry.get("contexts", [])

            if not question:
                continue
            if not answer:
                continue
            if not contexts or all(c.strip() == "" for c in contexts):
                continue
            if "i don't know based on the provided documents." in answer.lower():
                continue
            if "no relevant passages" in answer.lower():
                continue

            all_entries.append({
                "question":question,
                "answer":answer,
                "contexts":[str(c) for c in contexts if c],
                "ground_truth":"",
                "source":"real_log"
            })

    print(f"[INFO] Loaded {len(all_entries)} valid real log entries")
    return all_entries

def generate_synthetic_qa(pdf_path: Path, count: int = 20) -> list[dict]:
    if not pdf_path.exists():
        print(f"[WARN] PDF not found at {pdf_path} — skipping synthetic generation")
        return []

    print(f"[INFO] Loading PDF for synthetic generation: {pdf_path.name}")
    docs = doc_loader(str(pdf_path), "legal")
    chunks = split_docs(docs)

    if not chunks:
        print("[WARN] No chunks extracted from PDF")
        return []

    step = max(1, len(chunks) // count)
    selected = chunks[::step][:count]
    print(f"[INFO] Selected {len(selected)} chunks from {len(chunks)} total")

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2
    )

    synthetic_pairs = []

    for i, chunk in enumerate(selected):
        print(f"[INFO] Generating synthetic QA {i+1}/{len(selected)}...")

        prompt = f"""You are building an evaluation dataset for a legal document QA system.
                Given the following excerpt from a legal document, generate:
                1. A specific, answerable question that can only be answered using this text
                2. A complete, accurate reference answer based solely on this text

                Requirements:
                - The question must be specific (not vague like "what does this say?")
                - The answer must be factual and directly from the text
                - Do not make up information not present in the text
                - Return ONLY valid JSON, no markdown, no explanation

                Return this exact format:
                {{"question": "your question here", "answer": "complete reference answer here"}}

                Legal document excerpt:
                {chunk.page_content[:1200]}"""

        try:
            response = model.invoke(prompt)
            text = response.content.strip()

            text = text.replace("```json", "").replace("```", "").strip()

            qa = json.loads(text)

            if not qa.get("question") or not qa.get("answer"):
                print(f"  [SKIP] Empty question or answer for chunk {i+1}")
                continue

            synthetic_pairs.append({
                "question":qa["question"].strip(),
                "answer":qa["answer"].strip(),
                "contexts":[chunk.page_content],
                "ground_truth":qa["answer"].strip(),
                "source":"synthetic"
            })

            time.sleep(2)

        except json.JSONDecodeError:
            print(f"  [SKIP] Could not parse JSON for chunk {i+1}")
            continue
        except Exception as e:
            print(f"  [SKIP] Error on chunk {i+1}: {type(e).__name__}: {e}")
            time.sleep(5)
            continue

    print(f"[INFO] Generated {len(synthetic_pairs)} synthetic QA pairs")
    return synthetic_pairs

def build_eval_dataset(real_entries: list, synthetic_entries: list) -> list[dict]:
    combined = real_entries + synthetic_entries

    seen = set()
    unique = []
    for entry in combined:
        q = entry["question"].lower().strip()
        if q not in seen:
            seen.add(q)
            unique.append(entry)

    print(f"[INFO] Combined dataset: {len(unique)} unique entries "
          f"({len(real_entries)} real + {len(synthetic_entries)} synthetic)")

    with open(EVAL_DATASET, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved eval dataset to {EVAL_DATASET}")
    return unique

def run_ragas_evaluation(dataset: list[dict]) -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"[ERROR] RAGAS not installed properly: {e}")
        print("Run: pip install ragas datasets")
        return {}

    with_ground_truth = [e for e in dataset if e.get("ground_truth", "").strip()]
    without_ground_truth = [e for e in dataset if not e.get("ground_truth", "").strip()]

    print(f"[INFO] Entries with ground_truth: {len(with_ground_truth)}")
    print(f"[INFO] Entries without ground_truth: {len(without_ground_truth)}")
    print(f"[INFO] Starting RAGAS evaluation — this will take several minutes...")

    all_results = {}

    if with_ground_truth:
        print(f"\n[INFO] Running all 4 metrics on {len(with_ground_truth)} entries...")

        ds_a = Dataset.from_list([{
            "question":e["question"],
            "answer":e["answer"],
            "contexts":e["contexts"],
            "ground_truth":e["ground_truth"],
        } for e in with_ground_truth])

        try:
            result_a = evaluate(
                ds_a,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ]
            )
            all_results["with_ground_truth"] = {
                "sample_count":len(with_ground_truth),
                "faithfulness":round(float(result_a["faithfulness"]), 4),
                "answer_relevancy":round(float(result_a["answer_relevancy"]), 4),
                "context_precision":round(float(result_a["context_precision"]), 4),
                "context_recall":round(float(result_a["context_recall"]), 4),
            }
            print(f"[DONE] Group A scores: {all_results['with_ground_truth']}")
        except Exception as e:
            print(f"[ERROR] Group A evaluation failed: {e}")

    if without_ground_truth:
        print(f"\n[INFO] Running 3 metrics on {len(without_ground_truth)} real log entries...")

        ds_b = Dataset.from_list([{
            "question":e["question"],
            "answer":e["answer"],
            "contexts":e["contexts"],
        } for e in without_ground_truth])

        try:
            result_b = evaluate(
                ds_b,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                ]
            )
            all_results["without_ground_truth"] = {
                "sample_count":len(without_ground_truth),
                "faithfulness":round(float(result_b["faithfulness"]), 4),
                "answer_relevancy":round(float(result_b["answer_relevancy"]), 4),
                "context_precision":round(float(result_b["context_precision"]), 4),
            }
            print(f"[DONE] Group B scores: {all_results['without_ground_truth']}")
        except Exception as e:
            print(f"[ERROR] Group B evaluation failed: {e}")

    return all_results

def save_results(scores: dict):
    from datetime import datetime

    output = {
        "label":"baseline",
        "date":datetime.now().isoformat(),
        "description":"First RAGAS run — untuned pipeline",
        "pipeline": {
            "embedding":"gemini-embedding-2-preview",
            "llm":"gemini-2.5-flash-lite",
            "retrieval":"BM25 + MultiQuery Vector (70/30 ensemble)",
            "reranker":"cross-encoder/ms-marco-MiniLM-L-6-v2",
            "chunk_size":800,
            "chunk_overlap":100,
            "top_k":3,
        },
        "scores": scores,
        "targets": {
            "faithfulness":0.85,
            "answer_relevancy":0.80,
            "context_precision":0.70,
            "context_recall":0.70,
        }
    }

    out_path = RESULTS_DIR / "baseline_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Baseline scores saved to {out_path}")

    print("\n" + "="*50)
    print("RAGAS BASELINE RESULTS")
    print("="*50)
    for group, metrics in scores.items():
        print(f"\n{group.replace('_', ' ').upper()}")
        for k, v in metrics.items():
            if k == "sample_count":
                print(f"  Samples:             {v}")
            else:
                target = output["targets"].get(k, "—")
                status = "✓" if isinstance(v, float) and v >= target else "✗"
                print(f"{k:<22} {v:.4f}   target {target}  {status}")
    print("="*50)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("LegalRAG — RAGAS Evaluation Pipeline")
    print("="*50 + "\n")

    real_entries = load_real_logs()

    synthetic_entries = generate_synthetic_qa(PDF_PATH, count=SYNTHETIC_COUNT)

    if not real_entries and not synthetic_entries:
        print("[ERROR] No data to evaluate. Check your log files and PDF path.")
        sys.exit(1)

    dataset = build_eval_dataset(real_entries, synthetic_entries)

    if len(dataset) < 10:
        print(f"[WARN] Only {len(dataset)} entries — scores may not be reliable.")
        print("Target is 30-50 entries for a meaningful evaluation.")
    os.environ["RAGAS_MAX_WORKERS"] = "1"
    scores = run_ragas_evaluation(dataset)

    if not scores:
        print("[ERROR] Evaluation returned no scores.")
        sys.exit(1)

    save_results(scores)

    print("\n[DONE] Evaluation complete.")