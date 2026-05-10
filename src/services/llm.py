import os
from dotenv import load_dotenv

load_dotenv()


# ======================
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


# ✅ NEW: JSON instruction for structured mode
JSON_INSTRUCTION = """

Return ONLY JSON in this format:

{
  "common_issues": [],
  "frequent_questions": [],
  "key_themes": [],
  "sentiment_counts": {
    "positive": 0,
    "negative": 0,
    "question": 0,
    "neutral": 0
  }
}

Do NOT include any paragraph or explanation.
"""


import time

def call_llm(prompt, model, temperature, provider):

    # CI mode
    if os.getenv("CI"):
        return "dummy response"

    # Initialize client
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

    # 🔥 Retry loop
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            output = response.choices[0].message.content

            # 🔥 Handle empty response
            if not output or not output.strip():
                print(f"[LLM] Empty response (attempt {attempt+1}) → retrying...")
                time.sleep(2)
                continue

            return output

        except Exception as e:
            print(f"[LLM] ERROR (attempt {attempt+1}):", e)
            time.sleep(3)

    # 🔥 Final fallback
    print("[LLM] FAILED after retries")
    return ""


# ======================
# SUMMARY
# ======================

def generate_batch_summary(comments, model, provider, temperature, structured=False):

    print("\n[SUMMARY] Generating batch summary...")
    print(f"[SUMMARY] Comments count: {len(comments)}")

    comments_text = "\n".join(comments)

    prompt = SUMMARY_PROMPT_TEMPLATE.format(comments=comments_text)

    # ✅ ADD: structured control
    if structured:
        prompt += JSON_INSTRUCTION

    response = call_llm(prompt, model=model, temperature=temperature, provider=provider)
    print("[SUMMARY] Raw output preview:", response[:200])

    return response


# ======================
# CLASSIFICATION
# ======================

def classify_comments_batch(comments, model, provider):

    print("\n[CLASSIFY] Running classification batch...")
    print(f"[CLASSIFY] Total comments: {len(comments)}")

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


# ======================
# REPLIES
# ======================

def generate_replies_batch(
    comments,
    model,
    provider,
    temperature,
    video_context="",
    global_summary="",
    use_rag=False
):
    print("\n[REPLY] Generating replies...")
    print(f"[REPLY] Number of comments: {len(comments)}")

    formatted_comments = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(comments)]
    )

    prompt = REPLY_PROMPT_TEMPLATE.format(comments=formatted_comments)

    # ✅ ADD: RAG context control
    if use_rag:
        context_block = f"""
Video Context:
{video_context}

Audience Summary:
{global_summary}
"""
        prompt = context_block + prompt

    response = call_llm(prompt, model=model, temperature=temperature, provider=provider)

    lines = response.strip().split("\n")

    replies = []

    for line in lines:
        parts = line.split("|")
        if len(parts) == 2:
            replies.append(parts[1].strip())

    return replies

def generate_global_summary(batch_insights, model, provider):

    print("\n[GLOBAL SUMMARY] Generating final summary...")

    import json

    insights_text = json.dumps(batch_insights, indent=2)

    prompt = f"""
    You are analyzing aggregated insights from YouTube comments.

    Write a clear, well-structured summary in 4–5 short sentences.

    Guidelines:
    - Each sentence should represent ONE key idea:
    1. Overall sentiment
    2. Main strengths (praises)
    3. Key issues or gaps
    4. Common questions or expectations
    - Combine similar points into a single insight (do not list multiple similar items)
    - Focus only on patterns that appear repeatedly
    - Avoid mixing unrelated ideas in one sentence
    - Keep sentences concise and easy to read
    - Do NOT repeat information
    - Do NOT copy phrases directly from the input
    - Do NOT include numbers, percentages, or raw data
    - Do NOT mention JSON or formatting

    The output should feel like a clean human explanation, not a data dump.

    Insights:
    {insights_text}
    """

    response = call_llm(
        prompt,
        model=model,
        temperature=0,
        provider=provider
    )

    return response