"""Persistence and retrieval for indexed resumes: Supabase over HTTPS + two RPCs.

HTTPS rather than the psycopg pool because the Stage 0 audit found port 5432
unreachable from development machines while Supabase's REST API works everywhere
— and every other domain table here already goes through supabase-py.

This adapter is honest about failure: every operation either returns a real
result or raises `ResumeStoreError`. Whether a failure should stop the caller is a
decision for the caller, and the two callers decide differently:

- indexing an upload (`ingestion.index_resume`) turns it into an explicit error —
  a resume must never be reported indexed when it was not saved;
- interactive retrieval (`retrieval.retrieve_resume_evidence`) logs it and carries
  on with no evidence, because a conversation must never break over a resume.

Every call is scoped to (user_id, thread_id). There is no method that takes only
one of them.
"""

import asyncio
import logging
import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import NON_RETRIEVABLE_SECTIONS, ResumeSection

logger = logging.getLogger(__name__)

REPLACE_RPC = "replace_resume"
MATCH_RPC = "match_resume_chunks"
DOCUMENTS_TABLE = "resume_documents"

# Mirrors the clamp inside match_resume_chunks, so a bad k fails here, loudly,
# rather than being silently reduced by the database.
MAX_MATCH_COUNT = 20

# float32 is what pgvector stores, and 9 significant digits round-trip any float32
# exactly — shorter on the wire than repr() of a float64 with no loss.
_VECTOR_DIGITS = "{:.9g}"


class ResumeStoreError(Exception):
    """A store operation failed. The message carries no connection detail."""


class ChunkRecord(BaseModel):
    """One chunk as written: its embedded text and the vector of exactly that text."""

    model_config = ConfigDict(frozen=True)

    chunk_index: int = Field(ge=0)
    section: ResumeSection
    content: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    page: int | None = Field(default=None, ge=1)
    embedding: list[float]


class StoredResume(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    chunk_count: int
    created_at: datetime


class ResumeStatus(BaseModel):
    """What is on file for one conversation. Never includes resume text."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    embedding_model: str
    content_sha256: str
    created_at: datetime
    candidate_facts: dict[str, Any] | None = None


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    chunk_index: int
    section: ResumeSection
    content: str
    token_count: int
    page: int | None
    similarity: float


# ---------------------------------------------------------------------------
# Pure helpers — the payload contract, testable without a client
# ---------------------------------------------------------------------------


def check_scope(user_id: int, thread_id: str) -> None:
    """Both halves of the scope, every time. `bool` is excluded: True == 1."""
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("user_id must be a positive integer")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string")


def serialize_vector(vector: Sequence[float], dimensions: int) -> str:
    """pgvector's text form, "[f1,f2,...]", validated before it leaves the process.

    The database would reject a wrong length or a non-finite value too, but only
    after the whole upload had been sent; failing here names the actual problem.
    """
    if len(vector) != dimensions:
        raise ValueError(f"expected a {dimensions}-dimension vector, got {len(vector)}")
    values = [float(x) for x in vector]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("vector contains NaN or infinity")
    return "[" + ",".join(_VECTOR_DIGITS.format(x) for x in values) + "]"


def build_replace_payload(
    *,
    user_id: int,
    thread_id: str,
    filename: str,
    content_sha256: str,
    page_count: int,
    embedding_model: str,
    dimensions: int,
    candidate_facts: dict[str, Any] | None,
    chunks: Sequence[ChunkRecord],
) -> dict[str, Any]:
    check_scope(user_id, thread_id)
    if not chunks:
        raise ValueError("a resume needs at least one chunk")
    indices = [c.chunk_index for c in chunks]
    if len(set(indices)) != len(indices):
        raise ValueError("chunk_index values must be unique")
    return {
        "p_user_id": user_id,
        "p_thread_id": thread_id,
        "p_filename": filename,
        "p_content_sha256": content_sha256,
        "p_page_count": page_count,
        "p_embedding_model": embedding_model,
        "p_candidate_facts": candidate_facts,
        "p_chunks": [
            {
                "chunk_index": c.chunk_index,
                "section": c.section.value,
                "content": c.content,
                "token_count": c.token_count,
                "page": c.page,
                "embedding": serialize_vector(c.embedding, dimensions),
            }
            for c in chunks
        ],
    }


def build_match_payload(
    *, user_id: int, thread_id: str, query_embedding: Sequence[float], dimensions: int, k: int
) -> dict[str, Any]:
    check_scope(user_id, thread_id)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_MATCH_COUNT:
        raise ValueError(f"k must be between 1 and {MAX_MATCH_COUNT}")
    return {
        "p_user_id": user_id,
        "p_thread_id": thread_id,
        "p_query_embedding": serialize_vector(query_embedding, dimensions),
        "p_match_count": k,
    }


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def _default_client():
    """The real supabase-py client. Tests replace this; see the xbuddy conftest."""
    from integrations.supabase.supabase_client import get_supabase_client

    return get_supabase_client()


class ResumeStore:
    """Resume persistence scoped to one (user_id, thread_id) per call."""

    def __init__(self, client: Any | None = None, *, dimensions: int | None = None) -> None:
        self._client = client
        if dimensions is None:
            from core.llm import EMBEDDING_DIMENSIONS

            dimensions = EMBEDDING_DIMENSIONS
        self.dimensions = dimensions

    def _get_client(self) -> Any:
        return self._client if self._client is not None else _default_client()

    async def replace(
        self,
        *,
        user_id: int,
        thread_id: str,
        filename: str,
        content_sha256: str,
        page_count: int,
        embedding_model: str,
        candidate_facts: dict[str, Any] | None,
        chunks: Sequence[ChunkRecord],
    ) -> StoredResume:
        """Atomically replace this conversation's resume. Raises on any failure."""
        payload = build_replace_payload(
            user_id=user_id,
            thread_id=thread_id,
            filename=filename,
            content_sha256=content_sha256,
            page_count=page_count,
            embedding_model=embedding_model,
            dimensions=self.dimensions,
            candidate_facts=candidate_facts,
            chunks=chunks,
        )
        rows = await self._rpc(REPLACE_RPC, payload, action="replace", thread_id=thread_id)
        if len(rows) != 1:
            raise ResumeStoreError(f"replace_resume returned {len(rows)} rows, expected 1")
        row = rows[0]
        if row.get("chunk_count") != len(chunks):
            # The function checks this itself; a mismatch here means the deployed
            # function is not the one this adapter was written against.
            raise ResumeStoreError("replace_resume reported a different chunk count than was sent")
        try:
            return StoredResume.model_validate(row)
        except Exception as exc:
            raise ResumeStoreError("replace_resume returned an unexpected shape") from exc

    async def status(self, *, user_id: int, thread_id: str) -> ResumeStatus | None:
        """The resume on file for this conversation, or None. Raises on failure."""
        check_scope(user_id, thread_id)

        def query() -> Any:
            return (
                self._get_client()
                .table(DOCUMENTS_TABLE)
                .select(
                    "document_id:id,filename,page_count,chunk_count,embedding_model,"
                    "content_sha256,created_at,candidate_facts"
                )
                .eq("user_id", user_id)
                .eq("thread_id", thread_id)
                .limit(1)
                .execute()
            )

        rows = await self._execute(query, action="status", thread_id=thread_id)
        if not rows:
            return None
        try:
            return ResumeStatus.model_validate(rows[0])
        except Exception as exc:
            raise ResumeStoreError("resume status had an unexpected shape") from exc

    async def match(
        self, *, user_id: int, thread_id: str, query_embedding: Sequence[float], k: int
    ) -> list[RetrievedChunk]:
        """Top-k chunks by exact cosine similarity, Summary and Header excluded.

        Raises on failure.
        """
        payload = build_match_payload(
            user_id=user_id,
            thread_id=thread_id,
            query_embedding=query_embedding,
            dimensions=self.dimensions,
            k=k,
        )
        rows = await self._rpc(MATCH_RPC, payload, action="match", thread_id=thread_id)
        try:
            chunks = [RetrievedChunk.model_validate(row) for row in rows]
        except Exception as exc:
            raise ResumeStoreError("match_resume_chunks returned an unexpected shape") from exc
        blocked = sorted({c.section.value for c in chunks if c.section in NON_RETRIEVABLE_SECTIONS})
        if blocked:
            # The function excludes these. If one arrives anyway, the deployed
            # function is not the migrated one — say so rather than use it.
            logger.error(
                "resume match: %s chunk returned; are migrations 003 and 004 applied?",
                ", ".join(blocked),
            )
            chunks = [c for c in chunks if c.section not in NON_RETRIEVABLE_SECTIONS]
        return chunks[:k]

    # -- plumbing -----------------------------------------------------------

    async def _rpc(self, name: str, payload: dict, *, action: str, thread_id: str) -> list[dict]:
        return await self._execute(
            lambda: self._get_client().rpc(name, payload).execute(), action=action, thread_id=thread_id
        )

    async def _execute(self, call, *, action: str, thread_id: str) -> list[dict]:
        """Run a blocking supabase-py call off the event loop; normalise failures."""
        try:
            response = await asyncio.to_thread(call)
        except Exception as exc:
            # Credentials missing (ValueError from the client factory), network,
            # PostgREST errors. The exception is logged with its traceback server-
            # side; the raised error carries only what was being attempted.
            logger.exception("resume store: %s failed for thread %s", action, thread_id[:8])
            raise ResumeStoreError(f"resume {action} failed") from exc
        data = getattr(response, "data", None)
        if data is None:
            return []
        if not isinstance(data, list):
            raise ResumeStoreError(f"resume {action} returned a non-list result")
        return data
