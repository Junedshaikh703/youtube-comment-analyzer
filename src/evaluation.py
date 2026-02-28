from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


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