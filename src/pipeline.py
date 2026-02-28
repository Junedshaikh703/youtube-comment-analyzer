import pandas as pd
import numpy as np

from src.llm import (
    generate_summary,
    classify_comment,
    generate_reply
)

from src.evaluation import (
    compute_summary_similarity,
    compute_reply_similarity
)


def run_pipeline():

    df = pd.read_csv(
        "experiment_dataset_cleaned.csv"
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

        # REPLIES
        for comment in comments:

            label = classify_comment(comment)

            if label in ["QUESTION"]:

                reply = generate_reply(comment)
                r_sim = compute_reply_similarity(comment, reply)

                reply_scores.append(r_sim)

    print("\nAverage Summary Similarity:", np.mean(summary_scores))

    if reply_scores:
        print("Average Reply Similarity:", np.mean(reply_scores))