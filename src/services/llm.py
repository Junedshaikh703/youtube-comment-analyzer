import os
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


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
You are the creator of a YouTube video responding to viewer comments.

Generate a short reply (maximum 20 words) for EACH comment.

Your reply should:
- Sound friendly and supportive
- Address the viewer’s comment directly
- Provide brief guidance if the viewer asks a question
- Politely acknowledge issues or concerns

Keep replies concise and natural.

Return replies STRICTLY in this format:

1|reply text
2|reply text
3|reply text

Do not include emojis.
Do not add explanations.

Comments:
{comments}
"""


def call_llm(prompt, model, temperature, provider):

    import os

    # 👉 If running in CI → return dummy
    if os.getenv("CI"):
        return "dummy response"

    # 👉 Import only when needed
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    elif provider == "deepseek":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return response.choices[0].message.content


def generate_summary(comments , model , provider, temperature):

    comments_text = "\n".join(comments)

    prompt = SUMMARY_PROMPT_TEMPLATE.format(comments=comments_text)

    response = call_llm(prompt, model=model, temperature=temperature, provider=provider)

    return response


def classify_comments_batch(comments , model, provider):

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

    response = call_llm(prompt, model=model, temperature=0, provider=provider)

    lines = response.strip().split("\n")

    labels = []

    for line in lines:
        parts = line.split("|")
        if len(parts) == 2:
            labels.append(parts[1].strip())

    return labels


def generate_replies_batch(comments, model, provider, temperature):

    formatted_comments = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(comments)]
    )

    prompt = REPLY_PROMPT_TEMPLATE.format(comments=formatted_comments)

    response = call_llm(prompt, model=model, temperature=temperature, provider=provider)

    lines = response.strip().split("\n")

    replies = []

    for line in lines:
        parts = line.split("|")
        if len(parts) == 2:
            replies.append(parts[1].strip())

    return replies



