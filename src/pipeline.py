import pandas as pd
import numpy as np

from src.llm import (
    generate_summary,
    classify_comments_batch,
    generate_reply
)

from src.evaluation import (
    compute_summary_similarity,
    compute_reply_similarity
)

BATCH_SIZE = 30

# CREATE BATCHES
def create_batches(items, batch_size):
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def run_pipeline():

    df = pd.read_csv(
        "data/processed/experiment_dataset_cleaned.csv"
    )

    grouped = df.groupby("video_id")["comment_text"].apply(list)

    summary_scores = []
    reply_scores = []

    for video_id, comments in grouped.items():

        print(f"\nProcessing video: {video_id}")

        # SUMMARY
        summary = generate_summary(comments)
        sim = compute_summary_similarity(comments, summary)
        summary_scores.append(sim)

        print("Summary Similarity:", sim)

        
        # Batch classification (ONE API CALL)
        all_labels = []

        batches = create_batches(comments, BATCH_SIZE)

        for batch in batches:
            labels = classify_comments_batch(batch)
            all_labels.extend(labels)

        print(len(comments), len(all_labels))

        for comment, label in zip(comments, all_labels):

            if label in ["QUESTION", "NEGATIVE"]:
                reply = generate_reply(comment)

                r_sim = compute_reply_similarity(comment, reply)
                reply_scores.append(r_sim)

    print("\nAverage Summary Similarity:", np.mean(summary_scores))

    if reply_scores:
        print("Average Reply Similarity:", np.mean(reply_scores))