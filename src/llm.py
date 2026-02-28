import os
from groq import Groq
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Local zero-shot classifier for efficient comment classification
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli",
                      device=-1)  # -1 = CPU, 0 = GPU if available

# Local summarizer for efficient summary generation  
summarizer = pipeline("summarization",
                      model="facebook/bart-large-cnn",
                      device=-1)


def generate_summary(comments):
    """Generate summary using local model (fast & free)."""
    
    comments_text = "\n".join(comments[:50])  # Limit to first 50 comments to stay within token limit
    
    # Local summarization: split long text into chunks if needed
    max_length = min(150, len(comments_text) // 4)
    min_length = max(50, max_length // 3)
    
    try:
        summary = summarizer(comments_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        # Fallback: simple extractive summary if local summarizer fails
        sentences = comments_text.split(". ")
        return ". ".join(sentences[:3]) + "."


def classify_comment(comment):
    """Classify comment using local zero-shot classification (fast & free)."""
    
    candidate_labels = ["POSITIVE", "NEGATIVE", "QUESTION", "NEUTRAL"]
    
    result = classifier(comment, candidate_labels, multi_class=False)
    
    return result["labels"][0]


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