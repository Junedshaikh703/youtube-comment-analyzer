import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# PROMPTS
# ======================

SUMMARY_PROMPT_TEMPLATE = """
You are analyzing YouTube comments to provide feedback to the creator.

Write a concise and well-structured paragraph summary (3–5 sentences) that:

• Describes the overall sentiment
• Highlights major praise or appreciation themes
• Mentions recurring concerns or requests
• Notes frequently asked questions

Include one or two representative phrases from the comments when helpful,
but avoid excessive quoting or repetition.
Stay close to the wording used in the comments while maintaining clarity.

Do not invent information.

Comments:
{comments}
"""


REPLY_PROMPT_TEMPLATE = """
Generate a short (maximum 20 words), friendly and professional reply 
to the following YouTube comment.

Do not add unnecessary details.
Do not include emojis unless appropriate.

Comment:
{comment}
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




def generate_reply(comment , model , temperature):

    prompt = REPLY_PROMPT_TEMPLATE.format(comment=comment)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return response.choices[0].message.content.strip()
