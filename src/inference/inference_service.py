import json

from src.llm import (
    generate_summary,
    classify_comments_batch,
    generate_replies_batch
)

# load best model config once
with open("configs/best_model_config.json") as f:
    config = json.load(f)

MODEL = config["model_name"]
PROVIDER = config["provider"]


def analyze_comments(comments):

    # 1️⃣ summary
    summary = generate_summary(
        comments,
        model=MODEL,
        provider=PROVIDER,
        temperature=0
    )

    # 2️⃣ classification
    labels = classify_comments_batch(
        comments,
        model=MODEL,
        provider=PROVIDER
    )

    # 3️⃣ filter comments needing replies
    target_comments = []

    for comment, label in zip(comments, labels):

        if label in ["QUESTION", "NEGATIVE"]:
            target_comments.append(comment)

    # 4️⃣ generate replies
    replies = []

    if target_comments:

        replies = generate_replies_batch(
            target_comments,
            model=MODEL,
            provider=PROVIDER,
            temperature=0
        )

    return summary, replies