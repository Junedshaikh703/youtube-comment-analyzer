import pandas as pd
import numpy as np
import yaml
import mlflow
import dagshub

dagshub.init(
    repo_owner="Junedshaikh703",
    repo_name="youtube-comment-analyzer",
    mlflow=True
)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MODEL = params["llm"]["model_name"]
BATCH_SIZE = params["llm"]["batch_size"]
SUMMARY_TEMP = params["llm"]["summary_temperature"]
REPLY_TEMP = params["llm"]["reply_temperature"]



from src.llm import (
    generate_summary,
    classify_comments_batch,
    generate_reply,
    SUMMARY_PROMPT_TEMPLATE,
    REPLY_PROMPT_TEMPLATE
)

from src.evaluation import (
    compute_summary_similarity,
    compute_reply_similarity
)


# CREATE BATCHES
def create_batches(items, batch_size):
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def run_pipeline():

    mlflow.set_experiment("Prompt Experimentation")


    run_name = f"prompt_test_{MODEL}_bs{BATCH_SIZE}"
    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", MODEL)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("summary_temperature", SUMMARY_TEMP)
        mlflow.log_param("reply_temperature", REPLY_TEMP)

        mlflow.set_tag("experiment_phase", "prompt_experimentation")
        mlflow.set_tag("task", "summary_and_reply")

        mlflow.log_text(
            SUMMARY_PROMPT_TEMPLATE,
            "summary_prompt.txt"
        )

        mlflow.log_text(
            REPLY_PROMPT_TEMPLATE,
            "reply_prompt.txt"
        )

        df = pd.read_csv(
            "data/processed/experiment_dataset_cleaned.csv"
        )

        grouped = df.groupby("video_id")["comment_text"].apply(list)

        summary_scores = []
        reply_scores = []

        for video_id, comments in grouped.items():

            print(f"\nProcessing video: {video_id}")

            # SUMMARY
            summary = generate_summary(comments , model=MODEL , temperature=SUMMARY_TEMP)
            sim = compute_summary_similarity(comments, summary)
            summary_scores.append(sim)

            print("Summary Similarity:", sim)

            mlflow.log_text(
            summary,
            f"summaries/{video_id}.txt"
            )

            
            # Batch classification (ONE API CALL)
            all_labels = []

            batches = create_batches(comments, BATCH_SIZE)

            for batch in batches:
                labels = classify_comments_batch(batch , model=MODEL)
                all_labels.extend(labels)

            print(len(comments), len(all_labels))

            for comment, label in zip(comments, all_labels):

                if label in ["QUESTION", "NEGATIVE"]:
                    reply = generate_reply(comment , model=MODEL , temperature=REPLY_TEMP)

                    r_sim = compute_reply_similarity(comment, reply)
                    reply_scores.append(r_sim)

        avg_summary_similarity = float(np.mean(summary_scores))

        print("\nAverage Summary Similarity:", avg_summary_similarity)

        mlflow.log_metric(
            "avg_summary_similarity",
            avg_summary_similarity
        )

        if reply_scores:
            avg_reply_similarity = float(np.mean(reply_scores))

            print("Average Reply Similarity:", avg_reply_similarity)

            mlflow.log_metric(
                "avg_reply_similarity",
                avg_reply_similarity
            )