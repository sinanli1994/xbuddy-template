"""Indexing an upload: extract, chunk, embed, persist — or fail explicitly.

The upload path must never report a resume indexed unless every step succeeded.
Offline: a fake embedder and a fake store stand in for OpenAI and Supabase.
"""

from pathlib import Path

import pytest
from resume_pdf import image_only_pdf, text_pdf

from agents.xbuddy.resume import ingestion
from agents.xbuddy.resume.chunking import chunk_document
from agents.xbuddy.resume.extraction import extract_pdf_text, normalize_text
from agents.xbuddy.resume.ingestion import ResumeIndexingError, clean_filename, index_resume
from agents.xbuddy.resume.models import ResumeExtractionError
from agents.xbuddy.resume.store import ResumeStoreError, StoredResume
from agents.xbuddy.resume.tokens import count_tokens

CORPUS = Path(__file__).resolve().parents[3] / "evals" / "resume_retrieval" / "corpus"
RESUME = (CORPUS / "backend_to_ai.txt").read_text(encoding="utf-8")
D = 1536


class FakeEmbedder:
    def __init__(self, fail: Exception | None = None) -> None:
        self.batches: list[list[str]] = []
        self.fail = fail

    def __call__(self, texts):
        self.batches.append(list(texts))
        if self.fail:
            raise self.fail
        return [[float(i + 1)] + [0.0] * (D - 1) for i, _ in enumerate(texts)]


class FakeStore:
    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    async def replace(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return StoredResume(document_id="doc-123", chunk_count=len(kwargs["chunks"]),
                            created_at="2026-09-11T12:00:00+00:00")


async def index(data=None, **overrides):
    embedder = overrides.pop("embedder", FakeEmbedder())
    store = overrides.pop("store", FakeStore())
    kwargs = {"filename": "Jordan Avery CV.pdf", "user_id": 7, "thread_id": "thread-1"}
    kwargs.update(overrides)
    result = await index_resume(data or text_pdf(RESUME), store=store, embed_batch=embedder, **kwargs)
    return result, embedder, store


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_resume_is_indexed_end_to_end():
    result, _, store = await index()
    assert result.document_id == "doc-123"
    assert result.page_count == 2 and result.chunk_count == len(store.calls[0]["chunks"]) == 12
    assert result.embedding_model == "text-embedding-3-small"
    assert result.embedded_tokens > 0


@pytest.mark.asyncio
async def test_every_chunk_is_embedded_in_one_batch():
    _, embedder, store = await index()
    assert len(embedder.batches) == 1
    assert len(embedder.batches[0]) == len(store.calls[0]["chunks"])


@pytest.mark.asyncio
async def test_the_embedded_text_is_exactly_what_stage_2_evaluated():
    """The chunk's section-prefixed content, normalised — the representation whose
    retrieval quality the eval measured."""
    _, embedder, _ = await index()
    expected = [normalize_text(c.content) for c in chunk_document(extract_pdf_text(text_pdf(RESUME))).chunks]
    assert embedder.batches[0] == expected


@pytest.mark.asyncio
async def test_the_stored_text_is_exactly_the_embedded_text():
    """Retrieval returns what was stored; it must be what was scored."""
    _, embedder, store = await index()
    stored = [c.content for c in store.calls[0]["chunks"]]
    assert stored == embedder.batches[0]
    assert all(c.token_count == count_tokens(c.content) for c in store.calls[0]["chunks"])


@pytest.mark.asyncio
async def test_each_vector_goes_with_its_own_chunk():
    _, _, store = await index()
    chunks = store.calls[0]["chunks"]
    assert [c.embedding[0] for c in chunks] == [float(i + 1) for i in range(len(chunks))]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


@pytest.mark.asyncio
async def test_metadata_reaches_the_store():
    data = text_pdf(RESUME)
    _, _, store = await index(data, candidate_facts={"current_role": "Senior Backend Engineer"})
    call = store.calls[0]
    assert (call["user_id"], call["thread_id"]) == (7, "thread-1")
    assert call["content_sha256"] == extract_pdf_text(data).content_sha256
    assert call["embedding_model"] == "text-embedding-3-small"
    assert call["candidate_facts"] == {"current_role": "Senior Backend Engineer"}
    assert call["filename"] == "Jordan Avery CV.pdf"


# --------------------------------------------------------------------------
# Failures are explicit, and nothing is reported indexed
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unusable_pdf_fails_before_anything_is_embedded_or_stored():
    embedder, store = FakeEmbedder(), FakeStore()
    with pytest.raises(ResumeExtractionError) as caught:
        await index(image_only_pdf(), embedder=embedder, store=store)
    assert caught.value.code == "no_text"
    assert embedder.batches == [] and store.calls == []


@pytest.mark.asyncio
async def test_an_embedding_failure_fails_the_upload_and_stores_nothing():
    store = FakeStore()
    with pytest.raises(ResumeIndexingError) as caught:
        await index(embedder=FakeEmbedder(fail=RuntimeError("openai 503")), store=store)
    assert caught.value.code == "embedding_failed"
    assert "openai" not in caught.value.message.lower()
    assert store.calls == []


@pytest.mark.asyncio
async def test_a_persistence_failure_fails_the_upload():
    """The one that must never be quiet: embedded but not saved is NOT indexed."""
    with pytest.raises(ResumeIndexingError) as caught:
        await index(store=FakeStore(fail=ResumeStoreError("resume replace failed")))
    assert caught.value.code == "persistence_failed"


@pytest.mark.asyncio
async def test_a_malformed_embedding_response_fails_the_upload():
    class WrongDims(FakeEmbedder):
        def __call__(self, texts):
            return [[0.1] * 3 for _ in texts]

    with pytest.raises(ResumeIndexingError) as caught:
        await index(embedder=WrongDims())
    assert caught.value.code == "embedding_failed"


@pytest.mark.asyncio
async def test_too_many_chunks_is_refused_before_paying_for_embeddings(monkeypatch):
    monkeypatch.setattr(ingestion, "MAX_CHUNKS", 3)
    embedder = FakeEmbedder()
    with pytest.raises(ResumeIndexingError) as caught:
        await index(embedder=embedder)
    assert caught.value.code == "too_many_chunks"
    assert embedder.batches == []


@pytest.mark.asyncio
async def test_the_real_embedding_api_is_unreachable_in_tests(no_real_embedding_calls):
    """Without an injected embedder, indexing reaches the guarded seam — and fails
    as an explicit indexing error, which is the upload contract."""
    with pytest.raises(ResumeIndexingError) as caught:
        await index_resume(text_pdf(RESUME), filename="cv.pdf", user_id=7, thread_id="t", store=FakeStore())
    assert caught.value.code == "embedding_failed"
    assert no_real_embedding_calls == ["get_embeddings"]
    no_real_embedding_calls.clear()


# --------------------------------------------------------------------------
# Filenames
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("resume.pdf", "resume.pdf"),
        ("C:\\Users\\me\\Desktop\\resume.pdf", "resume.pdf"),
        ("/home/me/../etc/resume.pdf", "resume.pdf"),
        ("", "resume.pdf"),
        (None, "resume.pdf"),
        ("   ", "resume.pdf"),
    ],
)
def test_filenames_are_reduced_to_a_display_name(raw, clean):
    assert clean_filename(raw) == clean


def test_filenames_are_bounded():
    assert len(clean_filename("x" * 1000 + ".pdf")) == 255
