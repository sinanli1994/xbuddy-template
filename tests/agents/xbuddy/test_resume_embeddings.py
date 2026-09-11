"""Resume RAG embeddings: the seam, the network guard, cache keys, caching, and ranking.

Nothing here calls the embedding API. Embedding code under test receives a
deterministic double; the autouse guard in tests/conftest.py fails any test that
reaches the real client by any route.
"""

import asyncio
import hashlib
import math
import re

import pytest
from pydantic import SecretStr

from agents.xbuddy.resume.embeddings import (
    CachedEmbedder,
    EmbeddingCacheMiss,
    embedding_key,
    rank_by_cosine,
)

# Captured at import time — before the autouse guard replaces the module attribute —
# so the seam's own construction can still be tested. It is never used to embed.
from core.llm import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from core.llm import get_embeddings as real_get_embeddings
from core.settings import settings

DIMS = 32


class FakeEmbedder:
    """Deterministic bag-of-words hashing; records every batch it is sent."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * DIMS
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
            vectors.append(vector)
        return vectors

    @property
    def texts_sent(self) -> int:
        return sum(len(batch) for batch in self.batches)


def embedder(fake=None, **kwargs) -> CachedEmbedder:
    return CachedEmbedder(fake or FakeEmbedder(), model="test-model", dimensions=DIMS, **kwargs)


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------


@pytest.fixture
def fresh_seam():
    real_get_embeddings.cache_clear()
    yield
    real_get_embeddings.cache_clear()


def test_the_seam_is_text_embedding_3_small_at_1536(monkeypatch, fresh_seam):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr("sk-test-not-real"))
    client = real_get_embeddings()
    assert (EMBEDDING_MODEL, EMBEDDING_DIMENSIONS) == ("text-embedding-3-small", 1536)
    assert client.model == "text-embedding-3-small"
    assert client.dimensions == 1536
    assert client.openai_api_key.get_secret_value() == "sk-test-not-real"


def test_the_seam_is_shared_like_get_model(monkeypatch, fresh_seam):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr("sk-test-not-real"))
    assert real_get_embeddings() is real_get_embeddings()


def test_the_seam_refuses_to_build_without_a_key(monkeypatch, fresh_seam):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(Exception, match="api_key"):
        real_get_embeddings()


# --------------------------------------------------------------------------
# The guard — no test reaches the API
# --------------------------------------------------------------------------


def test_the_seam_is_blocked_during_tests(no_real_embedding_calls):
    from core.llm import get_embeddings

    with pytest.raises(AssertionError, match="Unmocked embedding call"):
        get_embeddings()
    no_real_embedding_calls.clear()


def test_a_directly_built_client_is_blocked_too(no_real_embedding_calls):
    """The route that matters: code that builds its own client, bypassing the seam."""
    from langchain_openai import OpenAIEmbeddings

    client = OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-test-not-real")
    with pytest.raises(AssertionError):
        client.embed_documents(["resume text"])
    with pytest.raises(AssertionError):
        client.embed_query("query")
    with pytest.raises(AssertionError):
        asyncio.run(client.aembed_documents(["resume text"]))
    assert no_real_embedding_calls == ["embed_documents", "embed_query", "aembed_documents"]
    no_real_embedding_calls.clear()


# --------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------


def test_keys_are_deterministic():
    assert embedding_key("Built APIs", model="m", dimensions=8) == embedding_key("Built APIs", model="m", dimensions=8)


def test_the_key_format_is_pinned():
    """Changing the format silently invalidates every cached vector. Pinned so that
    happens only on purpose."""
    expected = hashlib.sha256(b"m\x1f8\x1fBuilt APIs").hexdigest()
    assert embedding_key("Built APIs", model="m", dimensions=8) == expected


def test_layout_whitespace_does_not_change_the_key():
    assert embedding_key("Built   APIs\n", model="m", dimensions=8) == embedding_key("Built APIs", model="m", dimensions=8)


def test_model_and_dimensions_are_part_of_the_key():
    base = embedding_key("x", model="m", dimensions=8)
    assert embedding_key("x", model="other", dimensions=8) != base
    assert embedding_key("x", model="m", dimensions=16) != base


def test_different_text_gets_a_different_key():
    assert embedding_key("Built APIs", model="m", dimensions=8) != embedding_key("Built UIs", model="m", dimensions=8)


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_a_cold_cache_embeds_everything_in_one_batch():
    fake = FakeEmbedder()
    e = embedder(fake)
    e.embed(["alpha", "beta", "gamma"])
    assert fake.batches == [["alpha", "beta", "gamma"]]
    assert (e.stats.embedded, e.stats.cache_hits) == (3, 0)


def test_a_warm_cache_makes_no_calls():
    fake = FakeEmbedder()
    e = embedder(fake)
    first = e.embed(["alpha", "beta"])
    second = e.embed(["alpha", "beta"])
    assert fake.texts_sent == 2
    assert second == first
    assert e.stats.cache_hits == 2


def test_only_misses_are_sent_and_order_is_preserved():
    fake = FakeEmbedder()
    e = embedder(fake)
    e.embed(["alpha"])
    vectors = e.embed(["beta", "alpha", "gamma"])
    assert fake.batches[-1] == ["beta", "gamma"]
    assert vectors == FakeEmbedder()(["beta", "alpha", "gamma"])


def test_duplicates_in_one_call_are_embedded_once():
    fake = FakeEmbedder()
    embedder(fake).embed(["alpha", "alpha", "alpha  "])
    assert fake.batches == [["alpha"]]


def test_the_normalised_text_is_what_gets_embedded():
    fake = FakeEmbedder()
    embedder(fake).embed(["Built   the\t API\n\n\n\nfast"])
    assert fake.batches == [["Built the API\n\nfast"]]


def test_a_forbidden_miss_raises_without_sending_anything():
    fake = FakeEmbedder()
    e = embedder(fake, allow_calls=False)
    with pytest.raises(EmbeddingCacheMiss) as caught:
        e.embed(["alpha", "beta"])
    assert caught.value.missing == 2 and caught.value.tokens > 0
    assert fake.batches == []


def test_a_forbidden_embedder_still_serves_cached_texts():
    warm = embedder()
    warm.embed(["alpha"])
    cold = CachedEmbedder(FakeEmbedder(), model="test-model", dimensions=DIMS, cache=warm.cache, allow_calls=False)
    assert cold.embed(["alpha"]) == warm.embed(["alpha"])


def test_a_wrong_number_of_vectors_is_rejected():
    e = CachedEmbedder(lambda texts: [[0.0] * DIMS], model="m", dimensions=DIMS)
    with pytest.raises(ValueError, match="vectors for 2 texts"):
        e.embed(["a", "b"])


def test_a_wrong_dimensionality_is_rejected():
    e = CachedEmbedder(lambda texts: [[0.0] * 3 for _ in texts], model="m", dimensions=DIMS)
    with pytest.raises(ValueError, match="dimensions"):
        e.embed(["a"])


# --------------------------------------------------------------------------
# Cosine ranking
# --------------------------------------------------------------------------


def test_ranking_is_by_cosine_similarity():
    ranked = rank_by_cosine([1.0, 0.0], [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    assert [i for i, _ in ranked] == [2, 1, 0]
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[1][1] == pytest.approx(1 / math.sqrt(2))


def test_ranking_ignores_vector_length():
    """Cosine, not dot product: a longer chunk must not win by being longer."""
    assert rank_by_cosine([1.0, 0.0], [[10.0, 1.0], [1.0, 0.0]])[0][0] == 1


def test_ties_break_by_position():
    assert [i for i, _ in rank_by_cosine([1.0, 0.0], [[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]])] == [0, 1, 2]


def test_a_zero_vector_scores_zero_instead_of_failing():
    assert rank_by_cosine([1.0, 0.0], [[0.0, 0.0], [1.0, 0.0]]) == [(1, pytest.approx(1.0)), (0, 0.0)]


def test_no_candidates_ranks_nothing():
    assert rank_by_cosine([1.0], []) == []
