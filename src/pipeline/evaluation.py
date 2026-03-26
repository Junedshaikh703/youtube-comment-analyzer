from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def compute_structure_score(summary: str) -> float:
    """
    Computes structural quality score between 0 and 1.
    Penalizes excessive quoting and overly long summaries.
    """

    # Count quotation marks
    quote_count = summary.count('"')

    # Penalize too many quotes (max penalty 0.3)
    quote_penalty = min(quote_count * 0.03, 0.3)

    # Count sentences
    sentences = summary.split(".")
    sentence_count = len([s for s in sentences if s.strip() != ""])

    # Ideal range: 3–5 sentences
    if 3 <= sentence_count <= 5:
        length_penalty = 0
    else:
        length_penalty = 0.1

    structure_score = 1 - (quote_penalty + length_penalty)

    return max(structure_score, 0)


def compute_reply_constraint_score(reply: str) -> float:
    """
    Score reply quality based on simple constraints.
    Returns value between 0 and 1.
    """

    if not reply or reply.strip() == "":
        return 0.0

    score = 1.0

    # word length constraint
    word_count = len(reply.split())
    if word_count > 20:
        score -= 0.3

    # emoji check
    emoji_chars = ["🔥","😂","😍","👍","💯","❤️"]
    if any(e in reply for e in emoji_chars):
        score -= 0.1

    # repetition check (very simple)
    words = reply.lower().split()
    if len(words) != len(set(words)):
        score -= 0.1

    return max(score, 0)


def compute_summary_similarity(comments, summary):

    comment_embeddings = embedding_model.encode(comments)
    summary_embedding = embedding_model.encode([summary])

    mean_embedding = np.mean(comment_embeddings, axis=0).reshape(1, -1)

    similarity = cosine_similarity(mean_embedding, summary_embedding)[0][0]

    return similarity


def compute_reply_similarity(comment, reply):

    comment_emb = embedding_model.encode([comment])
    reply_emb = embedding_model.encode([reply])

    similarity = cosine_similarity(comment_emb, reply_emb)[0][0]

    return similarity