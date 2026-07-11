"""
qa_engine.py
--------------
A retrieval-based question answering engine. No external LLM call is
required: it builds a TF-IDF index over the document's chunks, finds
the passages most relevant to a question, and extracts the most
relevant sentence(s) from those passages as the answer.

This is intentionally transparent and citable — every answer points
back to the exact page(s) it came from, which matters more for
document Q&A than a fluent-but-unverifiable generated answer.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MIN_RELEVANCE = 0.08  # below this cosine similarity, treat as "not found"


class DocumentIndex:
    """Holds the TF-IDF index for one uploaded document's chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
        self.matrix = self.vectorizer.fit_transform(self.texts) if self.texts else None

    def search(self, query, top_k=3):
        """Return the top_k most relevant chunks for a query, with scores."""
        if self.matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        ranked = sorted(
            zip(self.chunks, scores), key=lambda x: x[1], reverse=True
        )
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]


def split_sentences(text):
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def extract_best_sentences(question, passage_text, max_sentences=2):
    """
    Within a passage, rank sentences by word-overlap with the question
    and return the top ones, in their original order.
    """
    question_words = set(re.findall(r"\b\w+\b", question.lower())) - STOPWORDS
    sentences = split_sentences(passage_text)

    if not sentences:
        return []

    scored = []
    for idx, sentence in enumerate(sentences):
        sentence_words = set(re.findall(r"\b\w+\b", sentence.lower()))
        overlap = len(question_words & sentence_words)
        scored.append((idx, sentence, overlap))

    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:max_sentences]
    top.sort(key=lambda x: x[0])  # restore original reading order

    return [s for _, s, score in top if score > 0] or [sentences[0]]


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "to", "does",
    "do", "how", "why", "when", "where", "who", "which", "this", "that",
    "explain", "tell", "me", "about", "can", "you", "please", "was", "were",
}


def answer_question(index: DocumentIndex, question, top_k=3):
    """
    Run retrieval + extraction for a question against a document index.

    Returns:
        dict: {
            "answer": str,
            "confidence": "high"|"medium"|"low"|"none",
            "sources": [{"page": int, "snippet": str, "score": float}, ...]
        }
    """
    results = index.search(question, top_k=top_k)

    if not results or results[0][1] < MIN_RELEVANCE:
        return {
            "answer": "I couldn't find anything relevant to that question in this document. "
                      "Try rephrasing, or ask about a topic covered in the PDF.",
            "confidence": "none",
            "sources": [],
        }

    top_chunk, top_score = results[0]
    best_sentences = extract_best_sentences(question, top_chunk["text"])
    answer_text = " ".join(best_sentences)

    if top_score >= 0.35:
        confidence = "high"
    elif top_score >= 0.18:
        confidence = "medium"
    else:
        confidence = "low"

    sources = [
        {
            "page": chunk["page"],
            "snippet": _snippet(chunk["text"]),
            "score": round(score, 3),
        }
        for chunk, score in results
        if score >= MIN_RELEVANCE
    ]

    return {
        "answer": answer_text,
        "confidence": confidence,
        "sources": sources,
    }


def _snippet(text, max_words=45):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"
