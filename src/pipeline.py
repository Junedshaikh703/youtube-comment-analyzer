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
PROVIDER = params["llm"]["provider"]
BATCH_SIZE = params["llm"]["batch_size"]
SUMMARY_TEMP = params["llm"]["summary_temperature"]
REPLY_TEMP = params["llm"]["reply_temperature"]



from src.llm import (
    generate_summary,
    classify_comments_batch,
    generate_replies_batch,
    SUMMARY_PROMPT_TEMPLATE,
    REPLY_PROMPT_TEMPLATE
)

from src.evaluation import (
    compute_summary_similarity,
    compute_reply_similarity,
    compute_structure_score,
    compute_reply_constraint_score
)


# CREATE BATCHES
def create_batches(items, batch_size):
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def run_pipeline():

    mlflow.set_experiment("Models Experimentation")


    run_name = f"model_test_{MODEL}"
    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", MODEL)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("summary_temperature", SUMMARY_TEMP)
        mlflow.log_param("reply_temperature", REPLY_TEMP)

        mlflow.set_tag("experiment_phase", "model_experimentation")
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
        cosine_scores = []
        structure_scores = []
        reply_cosine_scores = []
        reply_constraint_scores = []

        for video_id, comments in grouped.items():

            print(f"\nProcessing video: {video_id}")

            # SUMMARY
            summary = generate_summary(comments=comments[:45] , model=MODEL , provider=PROVIDER, temperature=SUMMARY_TEMP)


            cosine_score = compute_summary_similarity(comments, summary)
            structure_score = compute_structure_score(summary)

            final_score = (0.8 * cosine_score) + (0.2 * structure_score)

            summary_scores.append(final_score)

            # Optional: keep tracking individual metrics too
            cosine_scores.append(cosine_score)
            structure_scores.append(structure_score)
            

            print("Summary Similarity:", cosine_score)

            mlflow.log_text(
            summary,
            f"summaries/{video_id}.txt"
            )

            
            # Batch classification (ONE API CALL)
            all_labels = []


            batches = create_batches(comments, BATCH_SIZE)

            for batch in batches:
                labels = classify_comments_batch(comments=batch , model=MODEL , provider=PROVIDER)
                all_labels.extend(labels)

            print(len(comments), len(all_labels))

            # Collect comments needing replies
            target_comments = []
            target_indices = []

            for idx, (comment, label) in enumerate(zip(comments, all_labels)):
                if label in ["QUESTION", "NEGATIVE"]:
                    target_comments.append(comment)
                    target_indices.append(idx)

            # Generate replies in batch
            if target_comments:
                replies = generate_replies_batch(comments=target_comments, model=MODEL,provider=PROVIDER, temperature=REPLY_TEMP)

                for comment, reply in zip(target_comments, replies):

                    cosine_reply = compute_reply_similarity(comment, reply)
                    constraint_score = compute_reply_constraint_score(reply)

                    final_reply_score = (0.7 * cosine_reply) + (0.3 * constraint_score)

                    reply_scores.append(final_reply_score)
                    reply_cosine_scores.append(cosine_reply)
                    reply_constraint_scores.append(constraint_score)

                    if len(reply_scores) <= 5:
                        mlflow.log_text(
                            f"COMMENT: {comment}\nREPLY: {reply}",
                            f"reply_samples/{video_id}_{len(reply_scores)}.txt"
                        )
        
                

        # Compute averages
        avg_cosine = float(np.mean(cosine_scores))
        avg_structure = float(np.mean(structure_scores))
        avg_final = float(np.mean(summary_scores))

        print("\nAverage Cosine Similarity:", avg_cosine)
        print("Average Structure Score:", avg_structure)
        print("Average Final Score:", avg_final)

        # Log all metrics
        # mlflow.log_metric("avg_cosine_similarity", avg_cosine)
        # mlflow.log_metric("avg_structure_score", avg_structure)
        mlflow.log_metric("avg_final_score", avg_final)

        if reply_scores:
            avg_reply_cosine = float(np.mean(reply_cosine_scores))
            avg_reply_constraint = float(np.mean(reply_constraint_scores))
            avg_reply_final = float(np.mean(reply_scores))

            # mlflow.log_metric("avg_reply_cosine", avg_reply_cosine)
            # mlflow.log_metric("avg_reply_constraint", avg_reply_constraint)
            mlflow.log_metric("avg_reply_final", avg_reply_final)

            print("\nAverage Reply Cosine Similarity:", avg_reply_cosine)
            print("Average Reply Constraint Score:", avg_reply_constraint)      
            print("Average Reply Final Score:", avg_reply_final)