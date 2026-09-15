"""OPT-IN: the resume store against a real Supabase project. Skipped by default.

    RESUME_RAG_SUPABASE_INTEGRATION=1 uv run pytest tests/agents/xbuddy/test_resume_store_supabase.py -v

Requires migration 003 to be applied, and SUPABASE_URL + SUPABASE_SECRET_KEY. It
WRITES real rows, so it only ever touches conversations it invents
(`resume-rag-it-<uuid>`, user ids in 990200-990299) and deletes them all in a
`finally`, then checks nothing is left. No OpenAI call: vectors are synthetic, with
known geometry, so the expected ranking is exact.

The injected client bypasses the autouse `no_real_resume_store` guard on purpose —
this is the one test meant to reach the database.
"""

import os
import random
import uuid

import pytest
from postgrest.exceptions import APIError

from agents.xbuddy.resume.models import ResumeSection
from agents.xbuddy.resume.store import ChunkRecord, ResumeStore, ResumeStoreError, serialize_vector

pytestmark = pytest.mark.skipif(
    os.environ.get("RESUME_RAG_SUPABASE_INTEGRATION") != "1",
    reason="opt-in: set RESUME_RAG_SUPABASE_INTEGRATION=1 once migration 003 is applied",
)

D = 1536


def vec(**components: float) -> list[float]:
    v = [0.0] * D
    for key, value in components.items():
        v[int(key[1:])] = value
    return v


QUERY = vec(d0=1.0)


def record(i, section, content, vector):
    return ChunkRecord(chunk_index=i, section=section, content=content, token_count=5, page=1, embedding=vector)


@pytest.fixture
def live():
    from integrations.supabase.supabase_client import get_supabase_client

    client = get_supabase_client()
    run = uuid.uuid4().hex[:12]
    user = random.randint(990200, 990298)
    scopes = {
        "a": (user, f"resume-rag-it-{run}-a"),
        "b": (user, f"resume-rag-it-{run}-b"),      # same user, other thread
        "c": (user + 1, f"resume-rag-it-{run}-a"),  # other user, same thread id
    }
    yield ResumeStore(client), client, scopes
    for _, thread in scopes.values():
        client.table("resume_documents").delete().eq("thread_id", thread).execute()
    leftover = client.table("resume_chunks").select("id").like("thread_id", f"resume-rag-it-{run}-%").execute()
    assert leftover.data == [], "integration test left chunks behind"


async def put(store, scope, chunks, filename="it.pdf"):
    user, thread = scope
    return await store.replace(
        user_id=user, thread_id=thread, filename=filename, content_sha256="0" * 64, page_count=1,
        embedding_model="text-embedding-3-small", candidate_facts={"current_role": "IT fixture"}, chunks=chunks,
    )


@pytest.mark.asyncio
async def test_the_full_lifecycle_against_supabase(live):
    store, client, s = live

    first = await put(store, s["a"], [
        record(0, ResumeSection.HEADER, "[Header] fixture", vec(d0=0.5, d2=0.5)),
        record(1, ResumeSection.SUMMARY, "[Summary] perfect match, must be excluded", vec(d0=1.0)),
        record(2, ResumeSection.EXPERIENCE, "[Experience] best non-summary", vec(d0=0.9, d1=0.1)),
        record(3, ResumeSection.SKILLS, "[Skills] far", vec(d3=1.0)),
    ])
    await put(store, s["b"], [record(0, ResumeSection.EXPERIENCE, "[Experience] OTHER THREAD", vec(d0=1.0))])
    await put(store, s["c"], [record(0, ResumeSection.EXPERIENCE, "[Experience] OTHER USER", vec(d0=1.0))])

    # Retrieval: scoped, Summary excluded, exact cosine order.
    user, thread = s["a"]
    hits = await store.match(user_id=user, thread_id=thread, query_embedding=QUERY, k=3)
    assert [h.chunk_index for h in hits] == [2, 0, 3]
    assert all(h.section is not ResumeSection.SUMMARY for h in hits)
    assert all(h.document_id == first.document_id for h in hits)
    assert not any("OTHER" in h.content for h in hits)

    # Isolation, both directions.
    other = await store.match(user_id=s["b"][0], thread_id=s["b"][1], query_embedding=QUERY, k=3)
    assert [h.content for h in other] == ["[Experience] OTHER THREAD"]
    other_user = await store.match(user_id=s["c"][0], thread_id=s["c"][1], query_embedding=QUERY, k=3)
    assert [h.content for h in other_user] == ["[Experience] OTHER USER"]

    # Status.
    status = await store.status(user_id=user, thread_id=thread)
    assert status.document_id == first.document_id and status.chunk_count == 4
    assert status.candidate_facts == {"current_role": "IT fixture"}

    # Replacement: new document, old chunks gone, one document per conversation.
    second = await put(store, s["a"], [
        record(0, ResumeSection.PROJECTS, "[Projects] replaced", vec(d0=0.8, d1=0.2)),
    ], filename="it-v2.pdf")
    assert second.document_id != first.document_id
    old = client.table("resume_chunks").select("id").eq("document_id", first.document_id).execute()
    assert old.data == []
    docs = client.table("resume_documents").select("id").eq("user_id", user).eq("thread_id", thread).execute()
    assert [d["id"] for d in docs.data] == [second.document_id]

    # A failing replacement rolls back entirely: the previous resume survives.
    bad = [{"chunk_index": 0, "section": "experience", "content": "x", "token_count": 1, "page": 1,
            "embedding": serialize_vector([0.1] * (D - 1), D - 1)}]
    with pytest.raises(APIError, match="dimensions"):
        client.rpc("replace_resume", {
            "p_user_id": user, "p_thread_id": thread, "p_filename": "bad.pdf", "p_content_sha256": "0" * 64,
            "p_page_count": 1, "p_embedding_model": "text-embedding-3-small", "p_candidate_facts": None,
            "p_chunks": bad,
        }).execute()
    survivor = await store.status(user_id=user, thread_id=thread)
    assert survivor.document_id == second.document_id and survivor.filename == "it-v2.pdf"


def embedding_like(seed: int, boost: float = 0.0) -> list[float]:
    """Deterministic values shaped like real embeddings: small, dense, nine digits.

    A payload of all zeros serialises to two characters per value and would not
    test the size limit at all; these serialise the way real vectors do.
    """
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 0.0255) for _ in range(D)]
    v[0] += boost
    return v


def cap_sized_resume() -> list[ChunkRecord]:
    from agents.xbuddy.resume.ingestion import MAX_CHUNKS

    body = "• Delivered a measurable improvement to a production system with a named outcome. " * 8
    return [
        record(i, ResumeSection.EXPERIENCE, f"[Experience] chunk {i} {body}",
               embedding_like(i, boost=1.0 if i == 57 else 0.0))
        for i in range(MAX_CHUNKS)
    ]


@pytest.mark.asyncio
async def test_a_resume_at_the_chunk_cap_round_trips(live):
    """The upper bound the product allows (MAX_CHUNKS) through PostgREST in one call."""
    import json

    from agents.xbuddy.resume.store import build_replace_payload

    store, client, s = live
    user, thread = s["a"]
    chunks = cap_sized_resume()
    size = len(json.dumps(build_replace_payload(
        user_id=user, thread_id=thread, filename="cap.pdf", content_sha256="0" * 64, page_count=10,
        embedding_model="text-embedding-3-small", dimensions=D, candidate_facts=None, chunks=chunks,
    )))
    assert size > 1_000_000, f"payload only {size} bytes; not testing the upper bound"

    stored = await put(store, s["a"], chunks, filename="cap.pdf")
    assert stored.chunk_count == len(chunks)

    hits = await store.match(user_id=user, thread_id=thread, query_embedding=QUERY, k=3)
    assert hits[0].chunk_index == 57  # the one chunk boosted toward the query
    count = client.table("resume_chunks").select("id", count="exact").eq("document_id", stored.document_id).execute()
    assert count.count == len(chunks)


@pytest.mark.asyncio
async def test_a_cap_sized_upload_that_fails_late_leaves_nothing(live):
    """79 good chunks and a malformed last one, in one request. The function is one
    transaction, so the rows it had already inserted must not survive the failure."""
    from agents.xbuddy.resume.store import build_replace_payload

    store, client, s = live
    user, thread = s["a"]
    payload = build_replace_payload(
        user_id=user, thread_id=thread, filename="cap-bad.pdf", content_sha256="0" * 64, page_count=10,
        embedding_model="text-embedding-3-small", dimensions=D, candidate_facts=None, chunks=cap_sized_resume(),
    )
    payload["p_chunks"][-1]["embedding"] = serialize_vector([0.1] * (D - 1), D - 1)

    with pytest.raises(APIError, match="dimensions"):
        client.rpc("replace_resume", payload).execute()

    assert await store.status(user_id=user, thread_id=thread) is None
    partial = client.table("resume_chunks").select("id", count="exact").eq("thread_id", thread).execute()
    assert partial.count == 0, f"{partial.count} partial chunk rows survived a failed upload"


@pytest.mark.asyncio
async def test_a_retried_cap_sized_upload_does_not_duplicate(live):
    """A client that retries after a timeout sends the same upload twice. Replace
    semantics must leave exactly one document with exactly one set of chunks."""
    store, client, s = live
    user, thread = s["a"]
    chunks = cap_sized_resume()
    first = await put(store, s["a"], chunks, filename="cap.pdf")
    second = await put(store, s["a"], chunks, filename="cap.pdf")

    docs = client.table("resume_documents").select("id").eq("user_id", user).eq("thread_id", thread).execute()
    assert [d["id"] for d in docs.data] == [second.document_id]
    rows = client.table("resume_chunks").select("document_id", count="exact").eq("thread_id", thread).execute()
    assert rows.count == len(chunks)
    assert {r["document_id"] for r in rows.data} == {second.document_id}
    assert first.document_id != second.document_id


@pytest.mark.asyncio
async def test_a_conversation_without_a_resume(live):
    store, _, s = live
    user, thread = s["a"]
    assert await store.status(user_id=user, thread_id=thread) is None
    assert await store.match(user_id=user, thread_id=thread, query_embedding=QUERY, k=3) == []


@pytest.mark.asyncio
async def test_the_adapter_refuses_a_bad_vector_before_sending(live):
    store, _, s = live
    user, thread = s["a"]
    with pytest.raises(ValueError):
        await store.match(user_id=user, thread_id=thread, query_embedding=[0.1] * 5, k=3)
    with pytest.raises((ValueError, ResumeStoreError)):
        await store.match(user_id=user, thread_id=thread, query_embedding=QUERY, k=99)
