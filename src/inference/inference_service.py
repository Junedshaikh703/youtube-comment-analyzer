from __future__ import annotations
import os
import json

from src.services.yt_fetcher import (
    fetch_comments_paginated,
    fetch_video_metadata,
    fetch_transcript
)

from src.inference.parallel_processor import run_parallel_processing
from src.inference.rag_langchain import build_vector_store, retrieve_context
from src.services.llm import generate_global_summary, generate_replies_batch


def load_config() -> dict:
    if os.getenv("CI"):
        return {"model_name": "test-model", "provider": "groq", "reply_temperature": 0}

    with open("configs/best_model_config.json") as f:
        return json.load(f)


def update_aggregate(global_state: dict, chunk_results: list[dict]) -> dict:
    for chunk in chunk_results:
        insight = chunk.get("batch_insight", {})

        global_state["common_issues"].extend(insight.get("common_issues", []))
        global_state["frequent_questions"].extend(insight.get("frequent_questions", []))
        global_state["key_themes"].extend(insight.get("key_themes", []))

        for key in global_state["sentiment_counts"]:
            value = insight.get("sentiment_counts", {}).get(key, 0)

            try:
                value = int(value)  # 🔥 convert string → int
            except:
                value = 0  # fallback if something weird

            global_state["sentiment_counts"][key] += value

    return global_state


def dedup(lst: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in lst:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            out.append(item)
    return out


def analyze_comments(
    video_id: str,
    task_id: str,
    tasks: dict,
    max_comments: int = 1000,
    chunk_size: int = 50,
    max_workers: int = 2,
):

    config = load_config()
    model = config["model_name"]
    provider = config["provider"]
    reply_temp = config.get("reply_temperature", 0)

    # 🔥 SINGLE SOURCE OF TRUTH (SSOT)
    label_counts = {
        "positive": 0,
        "negative": 0,
        "question": 0,
        "neutral": 0,
    }

    global_state = {
        "common_issues": [],
        "frequent_questions": [],
        "key_themes": [],
        "sentiment_counts": {"positive": 0, "negative": 0, "question": 0, "neutral": 0},
    }

    target_comments = []
    seen_questions = set()   # 🔥 dedup
    total_fetched = 0

    # 🔁 STREAM LOOP
    for page_idx, page in enumerate(fetch_comments_paginated(video_id)):

        if total_fetched >= max_comments:
            break

        tasks[task_id]["progress"] = f"Processing page {page_idx + 1}"

        chunk_results = run_parallel_processing(
            all_comments=page,
            model=model,
            provider=provider,
            chunk_size=chunk_size,
            max_workers=max_workers,
        )

        # 🔥 aggregate ONLY for summary (not counts)
        global_state = update_aggregate(global_state, chunk_results)

        # 🔥 TRUE COUNTING FROM LABELS
        for chunk in chunk_results:

            comments = chunk.get("comments", [])
            labels = chunk.get("labels", [])

            if len(comments) != len(labels):
                print(f"[WARN] mismatch comments={len(comments)} labels={len(labels)}")
                continue

            for c, l in zip(comments, labels):

                l = l.lower()

                if l in label_counts:
                    label_counts[l] += 1

                # 🔥 build reply list from labels ONLY
                if l == "question":
                    if c not in seen_questions:
                        seen_questions.add(c)
                        target_comments.append(c)

        total_fetched += len(page)

        # 🔥 LIVE UI UPDATE (from labels ONLY)
        tasks[task_id]["counts"] = label_counts
        tasks[task_id]["processed"] = sum(label_counts.values())
        tasks[task_id]["total"] = sum(label_counts.values())

        # 🔥 PARTIAL SUMMARY (optional, text only)
        if total_fetched >= 100 and not tasks[task_id]["partial_summary"]:
            partial_summary = generate_global_summary(
                batch_insights=[global_state],
                model=model,
                provider=provider,
            )
            tasks[task_id]["partial_summary"] = partial_summary

    # 🔥 FINAL SUMMARY (text only)
    tasks[task_id]["progress"] = "Generating summary..."

    aggregated = {
        "common_issues": dedup(global_state["common_issues"]),
        "frequent_questions": dedup(global_state["frequent_questions"]),
        "key_themes": dedup(global_state["key_themes"]),
        "sentiment_counts": global_state["sentiment_counts"],  # only for prompt
    }

    global_summary = generate_global_summary(
        batch_insights=[aggregated],
        model=model,
        provider=provider,
    )

    tasks[task_id]["result"]["summary"] = global_summary

    # 🔥 RAG
    tasks[task_id]["progress"] = "Building context..."
    transcript = fetch_transcript(video_id)
    vector_store = build_vector_store(transcript) if transcript else None

    # 🔥 GUARANTEE consistency
    tasks[task_id]["counts"]["question"] = len(target_comments)

    # 🔥 STREAM REPLIES
    tasks[task_id]["progress"] = "Generating replies..."

    for idx, comment in enumerate(target_comments):

        context = retrieve_context(comment, vector_store) if vector_store else ""

        replies = generate_replies_batch(
            comments=[comment],
            model=model,
            provider=provider,
            temperature=reply_temp,
            video_context=context,
            global_summary=global_summary,
            use_rag=True
        )

        if replies:
            tasks[task_id]["result"]["replies"].append((comment, replies[0]))

        tasks[task_id]["progress"] = f"Replies: {idx+1}/{len(target_comments)}"

    tasks[task_id]["status"] = "completed"