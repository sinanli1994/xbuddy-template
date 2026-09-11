"""Index an uploaded resume: extract, chunk, embed, and persist — or fail loudly.

The upload path is the one place Resume RAG must never degrade quietly. Every step
either succeeds or raises, and nothing reports the resume as indexed unless
extraction, embedding, and the database write all succeeded:

    ResumeExtractionError  — the PDF itself is unusable       (endpoint: 4xx)
    ResumeIndexingError    — embedding or persistence failed  (endpoint: 5xx)

The text embedded for each chunk is exactly the text Stage 2 evaluated — the
chunk's section-prefixed `content`, normalised — and it is exactly the text
stored, so what retrieval later returns is what was scored.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Literal

from .chunking import chunk_document
from .embeddings import CachedEmbedder, EmbedBatch
from .extraction import extract_pdf_text, normalize_text
from .models import ResumeSection
from .store import ChunkRecord, ResumeStore, ResumeStoreError
from .tokens import count_tokens

logger = logging.getLogger(__name__)

# A ten-page resume is roughly 70 chunks at the eval's granularity. Past this the
# document is not a resume, and the replace payload (one vector per chunk) gets
# large enough to be worth refusing.
MAX_CHUNKS = 80
MAX_FILENAME = 255


class ResumeIndexingError(Exception):
    """Embedding or persistence failed; the resume is NOT indexed."""

    def __init__(
        self,
        code: Literal["no_chunks", "too_many_chunks", "embedding_failed", "persistence_failed"],
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class IndexedResume:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    sections: list[ResumeSection]
    embedding_model: str
    content_sha256: str
    embedded_tokens: int


def _default_embed_batch(texts: list[str]) -> list[list[float]]:
    from core.llm import get_embeddings

    return get_embeddings().embed_documents(texts)


def clean_filename(filename: str | None) -> str:
    """Just the base name, bounded. It is display metadata, never a path."""
    name = PureWindowsPath(filename or "").name.strip()  # handles both / and \
    return (name or "resume.pdf")[:MAX_FILENAME]


async def index_resume(
    data: bytes,
    *,
    filename: str | None,
    user_id: int,
    thread_id: str,
    candidate_facts: dict | None = None,
    store: ResumeStore | None = None,
    embed_batch: EmbedBatch | None = None,
) -> IndexedResume:
    """Extract, chunk, embed in one batch, and atomically replace the stored resume.

    `candidate_facts` are the unconfirmed Background values the upload endpoint
    extracts; they are stored alongside the resume and never written to user_data.
    """
    from core.llm import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

    document = extract_pdf_text(data)  # ResumeExtractionError propagates unchanged
    chunked = chunk_document(document)

    if not chunked.chunks:
        raise ResumeIndexingError("no_chunks", "We couldn't find any resume content in that PDF.")
    if len(chunked.chunks) > MAX_CHUNKS:
        raise ResumeIndexingError(
            "too_many_chunks", "That document is too long to be a resume. Please upload a shorter PDF."
        )

    contents = [normalize_text(chunk.content) for chunk in chunked.chunks]
    embedder = CachedEmbedder(
        embed_batch or _default_embed_batch, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS
    )
    try:
        vectors = await asyncio.to_thread(embedder.embed, contents)
    except Exception as exc:
        logger.exception("resume indexing: embedding failed for thread %s", thread_id[:8])
        raise ResumeIndexingError(
            "embedding_failed", "We couldn't process your resume right now. Please try again."
        ) from exc

    records = [
        ChunkRecord(
            chunk_index=chunk.index,
            section=chunk.section,
            content=content,
            token_count=count_tokens(content),
            page=chunk.page,
            embedding=vector,
        )
        for chunk, content, vector in zip(chunked.chunks, contents, vectors, strict=True)
    ]

    name = clean_filename(filename)
    store = store or ResumeStore()
    try:
        stored = await store.replace(
            user_id=user_id,
            thread_id=thread_id,
            filename=name,
            content_sha256=document.content_sha256,
            page_count=document.page_count,
            embedding_model=EMBEDDING_MODEL,
            candidate_facts=candidate_facts,
            chunks=records,
        )
    except ResumeStoreError as exc:
        raise ResumeIndexingError(
            "persistence_failed", "We couldn't save your resume. Please try again."
        ) from exc

    logger.info(
        "resume indexed: thread %s, %d chunks, %d pages",
        thread_id[:8],
        stored.chunk_count,
        document.page_count,
    )
    return IndexedResume(
        document_id=stored.document_id,
        filename=name,
        page_count=document.page_count,
        chunk_count=stored.chunk_count,
        sections=list(chunked.sections),
        embedding_model=EMBEDDING_MODEL,
        content_sha256=document.content_sha256,
        embedded_tokens=embedder.stats.embedded_tokens,
    )
