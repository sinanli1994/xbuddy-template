"""Interactive retrieval: top-k evidence, or nothing — never an exception.

This path runs inside a live JobBuddy turn, so every failure must degrade to "no
evidence" and a log line. Offline: fakes for the embedder and the store.
"""

import logging

import pytest

from agents.xbuddy.resume import retrieval
from agents.xbuddy.resume.retrieval import RESUME_TOP_K, retrieve_resume_evidence
from agents.xbuddy.resume.store import ResumeStoreError, RetrievedChunk

D = 1536


@pytest.fixture(autouse=True)
def empty_query_cache(monkeypatch):
    monkeypatch.setattr(retrieval, "_QUERY_CACHE", {})


class FakeEmbedder:
    def __init__(self, fail=None):
        self.calls: list[list[str]] = []
        self.fail = fail

    def __call__(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise self.fail
        return [[0.5] * D for _ in texts]


class FakeStore:
    def __init__(self, rows=None, fail=None):
        self.calls: list[dict] = []
        self.rows = rows if rows is not None else [evidence(2), evidence(5)]
        self.fail = fail

    async def match(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return self.rows


def evidence(i):
    return RetrievedChunk(id=f"c{i}", document_id="doc", chunk_index=i, section="experience",
                          content=f"[Experience] bullet {i}", token_count=5, page=1, similarity=0.8)


async def retrieve(query="Evidence of production deployment", **kwargs):
    embedder = kwargs.pop("embedder", FakeEmbedder())
    store = kwargs.pop("store", FakeStore())
    result = await retrieve_resume_evidence(7, "thread-1", query, store=store, embed_batch=embedder, **kwargs)
    return result, embedder, store


# --------------------------------------------------------------------------
# The locked production decision
# --------------------------------------------------------------------------


def test_the_default_top_k_is_three():
    assert RESUME_TOP_K == 3


@pytest.mark.asyncio
async def test_retrieval_defaults_to_k_three():
    _, _, store = await retrieve()
    assert store.calls[0]["k"] == 3


@pytest.mark.asyncio
async def test_an_explicit_k_is_propagated():
    _, _, store = await retrieve(k=5)
    assert store.calls[0]["k"] == 5


@pytest.mark.asyncio
async def test_the_scope_reaches_the_store_intact():
    _, _, store = await retrieve()
    assert (store.calls[0]["user_id"], store.calls[0]["thread_id"]) == (7, "thread-1")


@pytest.mark.asyncio
async def test_the_query_is_embedded_once_and_its_vector_is_used():
    result, embedder, store = await retrieve()
    assert embedder.calls == [["Evidence of production deployment"]]
    assert store.calls[0]["query_embedding"] == [0.5] * D
    assert [c.chunk_index for c in result] == [2, 5]


@pytest.mark.asyncio
async def test_a_repeated_query_is_not_re_embedded():
    """Section-level queries repeat every turn; paying for them once is the point."""
    embedder = FakeEmbedder()
    await retrieve(embedder=embedder)
    await retrieve(embedder=embedder)
    assert len(embedder.calls) == 1


@pytest.mark.asyncio
async def test_the_query_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(retrieval, "_QUERY_CACHE_LIMIT", 2)
    embedder = FakeEmbedder()
    for n in range(5):
        await retrieve(f"query {n}", embedder=embedder)
    assert len(retrieval._QUERY_CACHE) == 2


# --------------------------------------------------------------------------
# Degradation: never raise, always log
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_embedding_failure_returns_no_evidence(caplog):
    with caplog.at_level(logging.ERROR):
        result, _, store = await retrieve(embedder=FakeEmbedder(fail=RuntimeError("openai down")))
    assert result == [] and store.calls == []
    assert "continuing without evidence" in caplog.text


@pytest.mark.asyncio
async def test_a_store_failure_returns_no_evidence(caplog):
    with caplog.at_level(logging.ERROR):
        result, _, _ = await retrieve(store=FakeStore(fail=ResumeStoreError("resume match failed")))
    assert result == []
    assert "continuing without evidence" in caplog.text


@pytest.mark.asyncio
async def test_an_unexpected_error_also_returns_no_evidence():
    result, _, _ = await retrieve(store=FakeStore(fail=KeyError("anything at all")))
    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("user_id", "thread_id"), [(0, "thread-1"), (7, ""), (7, "   "), (True, "thread-1")])
async def test_an_incomplete_scope_retrieves_nothing_and_pays_nothing(user_id, thread_id):
    """Checked by the helper itself, before the embedding call — not left to the
    store, so it holds whichever store answers and never costs an API call."""
    embedder, store = FakeEmbedder(), FakeStore()
    result = await retrieve_resume_evidence(user_id, thread_id, "query", store=store, embed_batch=embedder)
    assert result == [] and embedder.calls == [] and store.calls == []


@pytest.mark.asyncio
async def test_no_resume_on_file_is_simply_no_evidence():
    result, _, _ = await retrieve(store=FakeStore(rows=[]))
    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", None])
async def test_a_blank_query_costs_nothing(query):
    result, embedder, store = await retrieve(query)
    assert result == [] and embedder.calls == [] and store.calls == []


@pytest.mark.asyncio
async def test_without_injection_the_real_services_are_unreachable_and_it_still_degrades(
    no_real_embedding_calls, no_real_resume_store
):
    """No fakes at all: the guards stop both real services, and the conversation
    still gets an empty result rather than an exception."""
    assert await retrieve_resume_evidence(7, "thread-1", "a new query nobody has asked") == []
    assert no_real_embedding_calls == ["get_embeddings"]
    no_real_embedding_calls.clear()
    no_real_resume_store.clear()
