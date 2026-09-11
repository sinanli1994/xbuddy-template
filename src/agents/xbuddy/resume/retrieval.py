"""Interactive resume retrieval: top-k evidence for a query, or nothing — never an error.

This is the path a live conversation takes, so it degrades rather than fails. No
resume, an embedding outage, an unreachable database, a malformed row: each one
is logged and returns an empty list, and JobBuddy carries on exactly as it would
for a user who never uploaded anything.

The production decision this encodes, fixed by the Stage 2 eval:
text-embedding-3-small · 1536 dims · exact cosine · Summary excluded (inside the
RPC) · top-3. The query goes through the same `embed_documents` path and the same
normalisation the eval used, so production ranks the way the eval measured.
"""

import asyncio
import logging

from .embeddings import CachedEmbedder, EmbedBatch, Vector
from .store import ResumeStore, RetrievedChunk, check_scope

logger = logging.getLogger(__name__)

RESUME_TOP_K = 3

# Section-level queries repeat constantly, so their vectors are kept in-process.
# Bounded, because a query built from user text could otherwise grow it forever.
_QUERY_CACHE: dict[str, Vector] = {}
_QUERY_CACHE_LIMIT = 512


def _default_embed_batch(texts: list[str]) -> list[list[float]]:
    from core.llm import get_embeddings

    return get_embeddings().embed_documents(texts)


def _trim_query_cache() -> None:
    excess = len(_QUERY_CACHE) - _QUERY_CACHE_LIMIT
    for key in list(_QUERY_CACHE)[: max(excess, 0)]:  # oldest first
        del _QUERY_CACHE[key]


async def retrieve_resume_evidence(
    user_id: int,
    thread_id: str,
    query: str,
    *,
    k: int = RESUME_TOP_K,
    store: ResumeStore | None = None,
    embed_batch: EmbedBatch | None = None,
) -> list[RetrievedChunk]:
    """The best `k` chunks of this conversation's resume for `query`. Never raises."""
    if not query or not query.strip():
        return []

    from core.llm import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

    try:
        # Before any embedding is paid for, and regardless of which store answers:
        # a query without a complete scope retrieves nothing.
        check_scope(user_id, thread_id)
        embedder = CachedEmbedder(
            embed_batch or _default_embed_batch,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            cache=_QUERY_CACHE,
        )
        [vector] = await asyncio.to_thread(embedder.embed, [query])
        _trim_query_cache()
        return await (store or ResumeStore()).match(
            user_id=user_id, thread_id=thread_id, query_embedding=vector, k=k
        )
    except Exception:
        # Deliberately broad: this runs inside a live turn. The traceback goes to
        # the server log; the conversation gets no evidence and continues.
        logger.exception("resume retrieval failed for thread %s; continuing without evidence", str(thread_id)[:8])
        return []
