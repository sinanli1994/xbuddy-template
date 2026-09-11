"""Retrievers under evaluation. Stage 1 ships only the free ones.

Every retriever ranks the chunks of **one** resume, because that is the production
shape: retrieval is always filtered to a single (user_id, thread_id) first, so the
candidate set is one person's resume and never the whole corpus.

`rank` returns chunk indices best-first and may return fewer than it was given. A
keyword retriever that shares no term with a chunk does not rank it at all — it
would otherwise be "retrieved" by document order, which rewards luck.

Stage 2 adds an embedding retriever behind the same interface.
"""

import math
import re
from collections import Counter
from typing import Protocol

# Small and deliberately generic: enough to stop "the", "and", "of" from dominating
# BM25 on a corpus of a dozen chunks, not a tuned list.
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your"]
)


def light_stem(token: str) -> str:
    """Strip the commonest English suffixes so "dashboards"/"dashboard" and
    "modelling"/"modelled" meet. Crude by design — a baseline that cannot even
    match plurals would make any embedding look good for the wrong reason."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            if suffix == "s" and token.endswith("ss"):
                return token
            return token[: -len(suffix)]
    return token


def terms(text: str) -> list[str]:
    return [light_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def rank(self, query: str, chunks: list[str]) -> list[int]: ...


class PositionalRetriever:
    """Ignores the query and returns chunks in document order — the chance floor.

    Any real retriever has to beat this; if one doesn't, the query set is not
    measuring retrieval at all.
    """

    name = "positional"

    def rank(self, query: str, chunks: list[str]) -> list[int]:
        return list(range(len(chunks)))


class EmbeddingRetriever:
    """Dense retrieval: cosine similarity between the query and each chunk's content.

    Ranks every candidate — there is no similarity cutoff, because a threshold is a
    separate decision the eval has not measured yet. Vectors come through a
    `CachedEmbedder`, so a warm cache makes this free and exactly repeatable.
    """

    name = "embedding"

    def __init__(self, embedder) -> None:  # CachedEmbedder; untyped to keep this module import-light
        self.embedder = embedder

    def rank(self, query: str, chunks: list[str]) -> list[int]:
        from agents.xbuddy.resume.embeddings import rank_by_cosine

        vectors = self.embedder.embed([query, *chunks])
        return [index for index, _ in rank_by_cosine(vectors[0], vectors[1:])]


class BM25Retriever:
    """Okapi BM25 over one resume's chunks, with light stemming.

    IDF is computed within the resume being searched, as a per-thread keyword
    search would see it. The +1 inside the log keeps IDF positive on a corpus this
    small, where a term in most chunks would otherwise score negative.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rank(self, query: str, chunks: list[str]) -> list[int]:
        docs = [Counter(terms(chunk)) for chunk in chunks]
        lengths = [sum(doc.values()) for doc in docs]
        average = (sum(lengths) / len(lengths)) if lengths else 0.0
        n = len(docs)

        scores: list[float] = []
        for doc, length in zip(docs, lengths, strict=True):
            score = 0.0
            for term in set(terms(query)):
                frequency = doc.get(term, 0)
                if not frequency:
                    continue
                df = sum(1 for other in docs if term in other)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                norm = frequency + self.k1 * (1 - self.b + self.b * length / (average or 1))
                score += idf * frequency * (self.k1 + 1) / norm
            scores.append(score)

        ranked = sorted(range(n), key=lambda i: (-scores[i], i))
        return [i for i in ranked if scores[i] > 0]
