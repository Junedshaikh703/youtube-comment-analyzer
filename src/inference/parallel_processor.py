"""
Runs preprocessing, classification, and batch summarization
concurrently using ThreadPoolExecutor.
Each "chunk" of comments is processed in parallel threads.
"""

from __future__ import annotations
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.services.llm import classify_comments_batch, generate_batch_summary


# ──────────────────────────────────────────────
# TEXT PREPROCESSING
# ──────────────────────────────────────────────

def preprocess_comment(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text


def preprocess_batch(comments: list[str]) -> list[str]:
    cleaned = [preprocess_comment(c) for c in comments]
    return [c for c in cleaned if len(c) > 2]


# ──────────────────────────────────────────────
# CHUNKING
# ──────────────────────────────────────────────

def create_chunks(items: list, chunk_size: int) -> list[list]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


# ──────────────────────────────────────────────
# 🔥 SAFE JSON PARSER (CORE FIX)
# ──────────────────────────────────────────────

def safe_parse_json(raw_text: str, chunk_index: int):

    fallback = {
        "common_issues": [],
        "frequent_questions": [],
        "key_themes": [],
        "sentiment_counts": {
            "positive": 0,
            "negative": 0,
            "question": 0,
            "neutral": 0,
        },
    }

    if not raw_text or not raw_text.strip():
        print(f"[CHUNK {chunk_index}] ⚠️ Empty LLM response")
        return fallback

    # 1️⃣ Try direct parse
    try:
        return json.loads(raw_text)
    except:
        pass

    # 2️⃣ Extract partial JSON
    try:
        match = re.search(r"\{.*", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON start found")

        cleaned = match.group()

        # Fix missing braces
        open_braces = cleaned.count("{")
        close_braces = cleaned.count("}")
        if open_braces > close_braces:
            cleaned += "}" * (open_braces - close_braces)

        # Fix missing brackets
        open_brackets = cleaned.count("[")
        close_brackets = cleaned.count("]")
        if open_brackets > close_brackets:
            cleaned += "]" * (open_brackets - close_brackets)

        return json.loads(cleaned)

    except Exception as e:
        print(f"[CHUNK {chunk_index}] ⚠️ JSON parsing failed:", e)
        print(f"[CHUNK {chunk_index}] 🔍 Raw output preview:", raw_text[:200])
        return fallback


# ──────────────────────────────────────────────
# CHUNK PROCESSING
# ──────────────────────────────────────────────

def process_chunk(
    chunk: list[str],
    chunk_index: int,
    model: str,
    provider: str,
) -> dict:

    print(f"\n[CHUNK {chunk_index}] 🚀 START")

    # 1️⃣ Preprocess
    cleaned = preprocess_batch(chunk)
    print(f"[CHUNK {chunk_index}] Preprocessed → {len(cleaned)} comments")

    if not cleaned:
        return {
            "chunk_index": chunk_index,
            "comments": [],
            "labels": [],
            "batch_insight": {
                "common_issues": [],
                "frequent_questions": [],
                "key_themes": [],
                "sentiment_counts": {
                    "positive": 0,
                    "negative": 0,
                    "question": 0,
                    "neutral": 0,
                },
            },
        }

    # 2️⃣ Classification
    print(f"[CHUNK {chunk_index}] 🔍 Classification...")
    labels = classify_comments_batch(
        comments=cleaned,
        model=model,
        provider=provider,
    )

    # 3️⃣ Batch summarization
    print(f"[CHUNK {chunk_index}] 🧠 Generating insight...")

    raw_insight = generate_batch_summary(
        comments=cleaned,
        model=model,
        temperature=0,
        provider=provider,
        structured=True
    )

    # 🔁 Retry once if weak response
    if not raw_insight or len(raw_insight.strip()) < 20:
        print(f"[CHUNK {chunk_index}] 🔁 Retrying LLM...")
        raw_insight = generate_batch_summary(
            comments=cleaned,
            model=model,
            temperature=0,
            provider=provider,
            structured=True
        )

    # 🔥 SAFE PARSE
    batch_insight = safe_parse_json(raw_insight, chunk_index)

    print(f"[CHUNK {chunk_index}] ✅ DONE")

    return {
        "chunk_index": chunk_index,
        "comments": cleaned,
        "labels": labels,
        "batch_insight": batch_insight,
    }


# ──────────────────────────────────────────────
# PARALLEL EXECUTION
# ──────────────────────────────────────────────

def run_parallel_processing(
    all_comments: list[str],
    model: str,
    provider: str,
    chunk_size: int = 50,
    max_workers: int = 2,
) -> list[dict]:

    print("\n[CHUNK] ⚙️ Parallel processing start")
    print(f"[CHUNK] Total comments: {len(all_comments)}")

    chunks = create_chunks(all_comments, chunk_size)
    print(f"[CHUNK] Total chunks: {len(chunks)}")

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_chunk, chunk, idx, model, provider): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[CHUNK {chunk_idx}] ✔ Completed")
            except Exception as exc:
                print(f"[CHUNK {chunk_idx}] ❌ ERROR:", exc)

    results.sort(key=lambda r: r["chunk_index"])

    print("[CHUNK] ✅ All chunks done\n")

    return results