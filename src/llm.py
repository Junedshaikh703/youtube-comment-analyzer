import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# PROMPTS
# ======================

SUMMARY_PROMPT_TEMPLATE = """
You are analyzing YouTube comments for a video.

Write a concise paragraph summary (4–6 sentences).

Your summary should capture:

• The overall sentiment of viewers
• The most common praises
• The most frequent complaints or issues
• The main questions viewers are asking

Focus only on themes that appear repeatedly.
Avoid mentioning rare or isolated comments.

Write clearly and professionally.
Do not invent information.
Do not include numbers or percentages.

Comments:
{comments}
"""


REPLY_PROMPT_TEMPLATE = """
You are a YouTube creator replying to viewer comments.

Generate a short reply (maximum 20 words) for EACH comment.

Guidelines:
- Be polite and professional
- Directly address the viewer’s comment
- If the viewer asks a question, provide a short helpful answer
- If the viewer reports a problem, acknowledge it respectfully
- Avoid unnecessary details

Return replies STRICTLY in this format:

1|reply text
2|reply text
3|reply text

Do not include emojis.
Do not add explanations.

Comments:
{comments}
"""



def generate_summary(comments , model , temperature):

    comments_text = "\n".join(comments)

    prompt = SUMMARY_PROMPT_TEMPLATE.format(comments=comments_text)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return response.choices[0].message.content


def classify_comments_batch(comments , model):

    formatted_comments = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(comments)]
    )

    prompt = f"""
You are a strict classifier.

Classify EACH comment into ONE label:
POSITIVE, NEGATIVE, QUESTION, NEUTRAL.

Return ONLY lines in EXACT format:
index|LABEL

Example:
1|POSITIVE
2|QUESTION

Comments:
{formatted_comments}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    lines = response.choices[0].message.content.strip().split("\n")

    labels = []

    for line in lines:
        parts = line.split("|")
        if len(parts) == 2:
            labels.append(parts[1].strip())

    return labels


def generate_replies_batch(comments, model, temperature):

    formatted_comments = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(comments)]
    )

    prompt = REPLY_PROMPT_TEMPLATE.format(comments=formatted_comments)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    lines = response.choices[0].message.content.strip().split("\n")

    replies = []

    for line in lines:
        parts = line.split("|")
        if len(parts) == 2:
            replies.append(parts[1].strip())

    return replies



