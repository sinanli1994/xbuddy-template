import os
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    with patch.dict(os.environ, {}, clear=True):
        yield


_EMBEDDING_METHODS = ("embed_documents", "embed_query", "aembed_documents", "aembed_query")


@pytest.fixture(autouse=True)
def no_real_embedding_calls(monkeypatch):
    """No test may reach the embedding API, whatever route it takes.

    Blocks both the `core.llm.get_embeddings` seam and the `OpenAIEmbeddings`
    methods themselves, at class level — so a caller that builds its own client, or
    imported the class before this fixture ran, is stopped too. A fake key would not
    be isolation: it would fail at the network, after the request was already built.

    Real embeddings happen only in the explicit eval command. Tests that exercise
    embedding code pass a deterministic double to `CachedEmbedder` instead.

    Yields the list of blocked attempts. Asserted empty at teardown, because code
    under test may catch the AssertionError; a test that trips the guard on purpose
    clears the list itself.
    """
    from langchain_openai import OpenAIEmbeddings

    attempts: list[str] = []

    def blocker(name: str):
        def blocked(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(
                f"Unmocked embedding call ({name}): supply a deterministic test double"
            )

        return blocked

    for method in _EMBEDDING_METHODS:
        monkeypatch.setattr(OpenAIEmbeddings, method, blocker(method))
    monkeypatch.setattr("core.llm.get_embeddings", blocker("get_embeddings"))
    yield attempts
    assert not attempts, f"A test attempted a real embedding call: {attempts}"
