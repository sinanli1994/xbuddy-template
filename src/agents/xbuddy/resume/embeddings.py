"""Embedding helpers for Resume RAG: stable cache keys, cached batching, cosine ranking.

Nothing here reaches the network on its own. `CachedEmbedder` is handed the batch
function to call — in production `get_embeddings().embed_documents`, in tests a
deterministic double — so the one place a paid call can start is explicit at the
call site.

Queries and chunks go through the same `embed_documents` path. OpenAI's
text-embedding-3 models are symmetric (`embed_query` is `embed_documents` of one
text), so one code path means one cache and no way for the two to drift.
"""

import hashlib
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .extraction import normalize_text
from .tokens import count_tokens

Vector = list[float]
EmbedBatch = Callable[[list[str]], list[Vector]]


def embedding_key(text: str, *, model: str, dimensions: int) -> str:
    """Stable cache key: the model, its dimensionality, and the normalised text.

    Dimensions are part of the key because the same model at 512 and at 1536
    dimensions produces different vectors. The text is normalised first, and it is
    the normalised text that gets embedded, so two inputs that differ only in
    layout whitespace share one key *and* one vector.
    """
    normalized = normalize_text(text)
    payload = f"{model}\x1f{dimensions}\x1f{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingCacheMiss(RuntimeError):
    """Texts are missing from the cache and this embedder may not call the API."""

    def __init__(self, missing: int, tokens: int) -> None:
        super().__init__(f"{missing} text(s) ({tokens} tokens) are not in the embedding cache")
        self.missing = missing
        self.tokens = tokens


@dataclass
class EmbeddingStats:
    """What an embedder has done: how much came from cache, how much was paid for."""

    requested: int = 0
    cache_hits: int = 0
    embedded: int = 0
    embedded_tokens: int = 0
    batches: int = 0
    embedded_keys: list[str] = field(default_factory=list)


class CachedEmbedder:
    """Embeds texts through a cache, calling `embed_batch` only for misses.

    - Misses are de-duplicated and sent in one batch per `embed` call.
    - The vector returned for a miss is the one read back from the cache after
      storing it, so a store that rounds on write (the eval's float32 disk cache)
      returns the same value on the first run as on every later one. Rankings
      cannot differ between a cold and a warm cache.
    - With `allow_calls=False`, a miss raises instead of spending money.
    """

    def __init__(
        self,
        embed_batch: EmbedBatch,
        *,
        model: str,
        dimensions: int,
        cache: MutableMapping[str, Vector] | None = None,
        allow_calls: bool = True,
    ) -> None:
        self._embed_batch = embed_batch
        self.model = model
        self.dimensions = dimensions
        self.cache: MutableMapping[str, Vector] = cache if cache is not None else {}
        self.allow_calls = allow_calls
        self.stats = EmbeddingStats()

    def key(self, text: str) -> str:
        return embedding_key(text, model=self.model, dimensions=self.dimensions)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        keys = [self.key(text) for text in texts]
        self.stats.requested += len(texts)

        # Keyed by cache key, so a text repeated in one call is embedded once.
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in self.cache:
                missing.setdefault(key, normalize_text(text))
        self.stats.cache_hits += sum(1 for key in keys if key not in missing)

        if missing:
            tokens = sum(count_tokens(text) for text in missing.values())
            if not self.allow_calls:
                raise EmbeddingCacheMiss(len(missing), tokens)
            vectors = self._embed_batch(list(missing.values()))
            if len(vectors) != len(missing):
                raise ValueError(f"embedder returned {len(vectors)} vectors for {len(missing)} texts")
            for key, vector in zip(missing, vectors, strict=True):
                if len(vector) != self.dimensions:
                    raise ValueError(f"expected {self.dimensions} dimensions, got {len(vector)}")
                self.cache[key] = list(vector)
            self.stats.embedded += len(missing)
            self.stats.embedded_tokens += tokens
            self.stats.batches += 1
            self.stats.embedded_keys.extend(missing)

        return [self.cache[key] for key in keys]


def rank_by_cosine(query: Vector, candidates: Sequence[Vector]) -> list[tuple[int, float]]:
    """Candidate indices with cosine similarity to `query`, best first.

    Ties break by index, so ranking is deterministic. A zero vector scores 0.0
    rather than dividing by zero. This is the same ordering pgvector's `<=>`
    (cosine distance = 1 - similarity) produces under an exact scan.
    """
    if not candidates:
        return []
    matrix = np.asarray(candidates, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    dots = matrix @ q
    scores = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [(i, float(scores[i])) for i in order]
