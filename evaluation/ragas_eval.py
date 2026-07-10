import json
import time
import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel

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

SYNTHETIC_COUNT = 25
MIN_REAL_ENTRIES = 25
MAX_DATASET_SIZE = 50

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

            if not question or not answer or not contexts:
                continue

            if all(c.strip() == "" for c in contexts):
                continue

            if "i don't know based on the provided documents." in answer.lower():
                continue

            if "no relevant passages" in answer.lower():
                continue

            all_entries.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": [str(c) for c in contexts if c],
                    "ground_truth": "",
                    "source": "real_log",
                }
            )

    print(f"[INFO] Loaded {len(all_entries)} valid real log entries")
    return all_entries


def generate_synthetic_qa(pdf_path: Path, count: int = SYNTHETIC_COUNT) -> list[dict]:
    if not pdf_path.exists():
        print(f"[WARN] PDF not found at {pdf_path}")
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

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
    )

    synthetic_pairs = []

    for index, chunk in enumerate(selected):
        print(f"[INFO] Generating synthetic QA {index + 1}/{len(selected)}...")

        prompt = f"""
        You are creating an evaluation dataset for a Legal RAG system.

        Given the following legal text, generate:

        1. One question
        2. One complete reference answer

        Rules:
        - Question must be answerable only using this text.
        - Answer must be completely supported by the text.
        - Do not hallucinate.
        - Return ONLY JSON.

        Format:

        {{
            "question": "...",
            "answer": "..."
        }}

        Legal Text:

        {chunk.page_content[:1200]}
        """

        try:
            response = llm.invoke(prompt)
            text = response.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            qa = json.loads(text)

            if not qa.get("question") or not qa.get("answer"):
                print(f"[SKIP] Invalid QA for chunk {index + 1}")
                continue

            synthetic_pairs.append(
                {
                    "question": qa["question"].strip(),
                    "answer": qa["answer"].strip(),
                    "contexts": [chunk.page_content],
                    "ground_truth": qa["answer"].strip(),
                    "source": "synthetic",
                }
            )
            time.sleep(2)

        except json.JSONDecodeError:
            print(f"[SKIP] JSON parsing failed for chunk {index + 1}")
        except Exception as e:
            print(f"[SKIP] {type(e).__name__}: {e}")
            time.sleep(5)

    print(f"[INFO] Generated {len(synthetic_pairs)} synthetic QA pairs")
    return synthetic_pairs


def build_eval_dataset(real_entries: list, synthetic_entries: list) -> list[dict]:
    combined = real_entries + synthetic_entries

    seen = set()
    dedup = []

    for entry in combined:
        q = entry["question"].lower().strip()
        if q not in seen:
            seen.add(q)
            dedup.append(entry)

    real = [e for e in dedup if e["source"] == "real_log"]
    synthetic = [e for e in dedup if e["source"] == "synthetic"]

    unique = synthetic[:SYNTHETIC_COUNT]
    remaining = MAX_DATASET_SIZE - len(unique)
    unique.extend(real[:remaining])

    print(
        f"[INFO] Combined dataset: {len(unique)} entries "
        f"({len(real[:remaining])} real + {len(unique)-len(real[:remaining])} synthetic)"
    )

    with open(EVAL_DATASET, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved eval dataset to {EVAL_DATASET}")
    return unique


def run_ragas_evaluation(dataset: list[dict]) -> dict:
    try:
        from google import genai
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
    except ImportError as e:
        print(f"[ERROR] Dependency missing: {e}")
        return {}

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    ragas_llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

    faithfulness_metric = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm)
    context_precision_metric = ContextPrecision(llm=ragas_llm)
    context_recall_metric = ContextRecall(llm=ragas_llm)

    class EvaluationResult(BaseModel):
        faithfulness: float | None = None
        answer_relevancy: float | None = None
        context_precision: float | None = None
        context_recall: float | None = None

    async def evaluate_row_full(row: dict) -> EvaluationResult:
        try:
            faith = await faithfulness_metric.ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
            )
            relevancy = await answer_relevancy_metric.ascore(
                user_input=row["question"],
                response=row["answer"],
            )
            precision = await context_precision_metric.ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
                reference=row["ground_truth"],
            )
            recall = await context_recall_metric.ascore(
                user_input=row["question"],
                retrieved_contexts=row["contexts"],
                reference=row["ground_truth"],
            )
            return EvaluationResult(
                faithfulness=getattr(faith, "value", 0.0),
                answer_relevancy=getattr(relevancy, "value", 0.0),
                context_precision=getattr(precision, "value", 0.0),
                context_recall=getattr(recall, "value", 0.0),
            )
        except Exception as e:
            print(f"[WARN] Error scoring a complete entry: {e}")
            return EvaluationResult()

    async def evaluate_row_no_gt(row: dict) -> EvaluationResult:
        try:
            faith = await faithfulness_metric.ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
            )
            relevancy = await answer_relevancy_metric.ascore(
                user_input=row["question"],
                response=row["answer"],
            )
            return EvaluationResult(
                faithfulness=getattr(faith, "value", 0.0),
                answer_relevancy=getattr(relevancy, "value", 0.0),
            )
        except Exception as e:
            print(f"[WARN] Error scoring a partial entry: {e}")
            return EvaluationResult()

    async def run_batch(rows: list[dict], with_gt: bool) -> list[EvaluationResult]:
        tasks = [evaluate_row_full(r) if with_gt else evaluate_row_no_gt(r) for r in rows]
        return await asyncio.gather(*tasks)

    with_ground_truth = [e for e in dataset if e.get("ground_truth", "").strip()]
    without_ground_truth = [e for e in dataset if not e.get("ground_truth", "").strip()]

    print(f"[INFO] Entries with ground_truth: {len(with_ground_truth)}")
    print(f"[INFO] Entries without ground_truth: {len(without_ground_truth)}")

    all_results = {}

    # ---------------- GROUP A (WITH GROUND TRUTH) ---------------- #
    if with_ground_truth:
        print(f"\n[INFO] Running all 4 metrics on {len(with_ground_truth)} entries...")
        try:
            results_a = asyncio.run(run_batch(with_ground_truth, with_gt=True))
            df = pd.DataFrame([r.model_dump() for r in results_a])
            scores = df.mean(numeric_only=True)

            all_results["with_ground_truth"] = {
                "sample_count": len(with_ground_truth),
                "faithfulness": round(float(scores.get("faithfulness", 0)), 4),
                "answer_relevancy": round(float(scores.get("answer_relevancy", 0)), 4),
                "context_precision": round(float(scores.get("context_precision", 0)), 4),
                "context_recall": round(float(scores.get("context_recall", 0)), 4),
            }
            print(f"[DONE] Group A scores: {all_results['with_ground_truth']}")
        except Exception as e:
            print(f"[ERROR] Group A evaluation failed: {type(e).__name__}: {e}")

    # ---------------- GROUP B (WITHOUT GROUND TRUTH) ---------------- #
    if without_ground_truth:
        print(f"\n[INFO] Running 2 metrics on {len(without_ground_truth)} entries...")
        try:
            results_b = asyncio.run(run_batch(without_ground_truth, with_gt=False))
            df = pd.DataFrame([r.model_dump() for r in results_b])
            scores = df.mean(numeric_only=True)

            all_results["without_ground_truth"] = {
                "sample_count": len(without_ground_truth),
                "faithfulness": round(float(scores.get("faithfulness", 0)), 4),
                "answer_relevancy": round(float(scores.get("answer_relevancy", 0)), 4),
            }
            print(f"[DONE] Group B scores: {all_results['without_ground_truth']}")
        except Exception as e:
            print(f"[ERROR] Group B evaluation failed: {type(e).__name__}: {e}")

    return all_results


def save_results(scores: dict):
    output = {
        "label": "baseline",
        "date": datetime.now().isoformat(),
        "description": "First RAGAS run — untuned pipeline",
        "pipeline": {
            "embedding": "gemini-embedding-2-preview",
            "llm": "gemini-2.0-flash",
            "retrieval": "BM25 + MultiQuery Vector (70/30 ensemble)",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "top_k": 3,
        },
        "scores": scores,
        "targets": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.70,
            "context_recall": 0.70,
        },
    }

    out_path = RESULTS_DIR / "baseline_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Baseline scores saved to {out_path}")
    print("\n" + "=" * 50)
    print("RAGAS BASELINE RESULTS")
    print("=" * 50)

    for group, metrics in scores.items():
        print(f"\n{group.replace('_', ' ').upper()}")
        for k, v in metrics.items():
            if k == "sample_count":
                print(f"  Samples:             {v}")
            else:
                target = output["targets"].get(k, 0.0)
                status = "✓" if isinstance(v, (int, float)) and v >= target else "✗"
                print(f"{k:<22} {v:.4f}   target {target}  {status}")
    print("=" * 50)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("LegalRAG — RAGAS Evaluation Pipeline")
    print("=" * 50 + "\n")

    real_entries = load_real_logs()
    synthetic_entries = generate_synthetic_qa(PDF_PATH, count=SYNTHETIC_COUNT)

    if not real_entries and not synthetic_entries:
        print("[ERROR] No data to evaluate.")
        sys.exit(1)

    dataset = build_eval_dataset(real_entries, synthetic_entries)

    if len(dataset) < MAX_DATASET_SIZE:
        print(f"[WARN] Only {len(dataset)} entries. Target is {MAX_DATASET_SIZE} entries.")

    scores = run_ragas_evaluation(dataset)

    if not scores:
        print("[ERROR] Evaluation failed.")
        sys.exit(1)

    save_results(scores)
    print("\n[DONE] Evaluation complete.")