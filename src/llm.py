import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_summary(comments):

    comments_text = "\n".join(comments)

    prompt = f"""
The following are YouTube comments for a video.

Write a concise but comprehensive paragraph summary (3–7 sentences) in English.
The summary should clearly reflect:
- The overall sentiment of the comments
- The main themes of praise (if any)
- The main complaints or concerns (if any)
- Any common questions or recurring topics

Do not invent information.
Do not include percentages or numbers.
Keep the tone neutral and professional.

Comments:
{comments_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


def classify_comment(comment):

    prompt = f"""
Classify the following YouTube comment into one of the categories:

POSITIVE
NEGATIVE
QUESTION
NEUTRAL

Return only one word.


Comment: {comment}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def generate_reply(comment):

    prompt = f"""
Generate a short (maximum 20 words), friendly and professional reply 
to the following YouTube comment.

Do not add unnecessary details.
Do not include emojis unless appropriate.


Comment:
{comment}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()