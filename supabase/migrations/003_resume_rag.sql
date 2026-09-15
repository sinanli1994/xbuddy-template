-- ============================================================================
-- Phase 8 — Resume RAG: resume documents, embedded chunks, and two RPCs
--
-- Retrieval design, fixed by the labelled eval in evals/resume_retrieval/:
--   section-aware chunks · text-embedding-3-small · 1536 dimensions · cosine
--   Summary excluded at retrieval time · top-3 by default
--
-- Follows the 002 conventions: IF NOT EXISTS everywhere, CREATE OR REPLACE only
-- for this migration's own functions, no DROP, safe to re-run.
--
-- Scope. One active resume per conversation, keyed by (user_id, thread_id).
-- thread_id is part of every filter: user_id is a small self-asserted integer in
-- this demo, while thread_id is an unguessable UUID, so scoping by user_id alone
-- would let a guessed integer pull someone else's resume into a conversation.
--
-- Access. Backend only. RLS is enabled with no policies, anon and authenticated
-- have no privileges on anything here, and the backend's secret key (service_role)
-- is granted exactly what the adapter uses. Resumes are personal data; the
-- browser never reaches these tables or functions.
--
-- No vector index. Every query filters to one conversation's ~10-20 chunks
-- first, and an exact scan over those is both instant and correct. An HNSW or
-- IVFFlat index would make the result approximate under a filter this selective
-- and could return fewer than k rows.
--
-- The raw PDF is never stored — only its SHA-256, page count, and the chunks.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.resume_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    thread_id TEXT NOT NULL CHECK (thread_id <> ''),
    filename TEXT NOT NULL,
    -- SHA-256 of the uploaded bytes: identifies the file, not its text.
    content_sha256 TEXT NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    page_count INTEGER NOT NULL CHECK (page_count >= 1),
    chunk_count INTEGER NOT NULL CHECK (chunk_count >= 1),
    -- Vectors from different models are not comparable. Recorded so a future
    -- model change can find and re-index what was embedded with the old one.
    embedding_model TEXT NOT NULL,
    -- Unconfirmed Background facts extracted from the resume. Never confirmed
    -- user state: JobBuddy proposes them and the user confirms or corrects.
    candidate_facts JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    -- One active resume per conversation; replace_resume enforces the swap.
    CONSTRAINT resume_documents_user_thread_key UNIQUE (user_id, thread_id)
);

CREATE TABLE IF NOT EXISTS public.resume_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.resume_documents(id) ON DELETE CASCADE,
    -- Denormalised from the document so retrieval filters without a join.
    user_id INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    -- The chunker's canonical section. NOT NULL and closed on purpose: the
    -- retrieval filter excludes 'summary', and a NULL or a 'Summary' spelled
    -- differently would slip past it silently. Must match ResumeSection in
    -- src/agents/xbuddy/resume/models.py (a test pins the two together).
    section TEXT NOT NULL CHECK (section IN (
        'header', 'summary', 'experience', 'projects', 'education', 'skills',
        'certifications', 'publications', 'awards', 'volunteering', 'other'
    )),
    -- Exactly the text that was embedded, section prefix included.
    content TEXT NOT NULL CHECK (content <> ''),
    token_count INTEGER NOT NULL CHECK (token_count >= 1),
    page INTEGER CHECK (page IS NULL OR page >= 1),
    embedding extensions.vector(1536) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    CONSTRAINT resume_chunks_document_index_key UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_resume_chunks_user_thread
    ON public.resume_chunks (user_id, thread_id);

-- ---------------------------------------------------------------------------
-- Access: backend only
-- ---------------------------------------------------------------------------

ALTER TABLE public.resume_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.resume_chunks ENABLE ROW LEVEL SECURITY;

-- Deliberately no policies. With RLS on and none defined, any role subject to
-- RLS sees and changes nothing; service_role bypasses RLS.

-- Revoke from all three API roles first, service_role included. This project's
-- default privileges grant TRUNCATE, REFERENCES, TRIGGER and MAINTAIN to anon,
-- authenticated and service_role on every table postgres creates in public
-- (production pg_default_acl: `service_role=Dxtm/postgres`). Granting on top of
-- those would leave service_role with privileges nothing here uses; revoking
-- everything and then granting exactly what the adapter needs does not.
REVOKE ALL ON TABLE public.resume_documents FROM anon, authenticated, service_role;
REVOKE ALL ON TABLE public.resume_chunks FROM anon, authenticated, service_role;

-- Exactly what the adapter needs. No UPDATE: a resume is replaced, never edited.
GRANT SELECT, INSERT, DELETE ON TABLE public.resume_documents TO service_role;
GRANT SELECT, INSERT, DELETE ON TABLE public.resume_chunks TO service_role;

-- ---------------------------------------------------------------------------
-- RPC: replace_resume — swap the conversation's resume atomically
-- ---------------------------------------------------------------------------
--
-- One PostgREST call is one transaction, so delete-old + insert-new either both
-- happen or neither does. A re-upload whose chunks fail to insert (a malformed
-- vector, a missing field) rolls back the delete too, and the previous resume
-- stays exactly as it was — never half old, half new.
--
-- p_chunks is a JSON array of
--   {"chunk_index": int, "section": text, "content": text,
--    "token_count": int, "page": int|null, "embedding": "[f1,...,f1536]"}
-- with the vector as pgvector's text form. The cast to vector(1536) rejects a
-- wrong dimension count, and pgvector rejects NaN and infinity.

CREATE OR REPLACE FUNCTION public.replace_resume(
    p_user_id INTEGER,
    p_thread_id TEXT,
    p_filename TEXT,
    p_content_sha256 TEXT,
    p_page_count INTEGER,
    p_embedding_model TEXT,
    p_candidate_facts JSONB,
    p_chunks JSONB
)
RETURNS TABLE (document_id UUID, chunk_count INTEGER, created_at TIMESTAMP WITH TIME ZONE)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
DECLARE
    v_document_id UUID;
    v_created_at TIMESTAMP WITH TIME ZONE;
    v_expected INTEGER;
    v_inserted INTEGER;
BEGIN
    IF p_user_id IS NULL OR p_thread_id IS NULL OR p_thread_id = '' THEN
        RAISE EXCEPTION 'replace_resume: user_id and thread_id are required';
    END IF;
    IF p_chunks IS NULL OR jsonb_typeof(p_chunks) <> 'array' OR jsonb_array_length(p_chunks) = 0 THEN
        RAISE EXCEPTION 'replace_resume: at least one chunk is required';
    END IF;
    v_expected := jsonb_array_length(p_chunks);

    -- Serialise concurrent uploads for the same conversation. Without this, two
    -- racing replacements could both delete and then collide on the unique key;
    -- with it, the later one waits and cleanly replaces the earlier.
    PERFORM pg_advisory_xact_lock(hashtextextended(p_user_id::TEXT || ':' || p_thread_id, 0));

    DELETE FROM public.resume_documents d
     WHERE d.user_id = p_user_id AND d.thread_id = p_thread_id;  -- chunks cascade

    INSERT INTO public.resume_documents AS d (
        user_id, thread_id, filename, content_sha256, page_count, chunk_count,
        embedding_model, candidate_facts
    )
    VALUES (
        p_user_id, p_thread_id, p_filename, p_content_sha256, p_page_count, v_expected,
        p_embedding_model, p_candidate_facts
    )
    RETURNING d.id, d.created_at INTO v_document_id, v_created_at;

    INSERT INTO public.resume_chunks (
        document_id, user_id, thread_id, chunk_index, section, content, token_count, page, embedding
    )
    SELECT
        v_document_id,
        p_user_id,
        p_thread_id,
        (c ->> 'chunk_index')::INTEGER,
        c ->> 'section',
        c ->> 'content',
        (c ->> 'token_count')::INTEGER,
        (c ->> 'page')::INTEGER,
        (c ->> 'embedding')::extensions.vector(1536)
    FROM jsonb_array_elements(p_chunks) AS c;

    GET DIAGNOSTICS v_inserted = ROW_COUNT;
    IF v_inserted <> v_expected THEN
        RAISE EXCEPTION 'replace_resume: inserted % of % chunks', v_inserted, v_expected;
    END IF;

    RETURN QUERY SELECT v_document_id, v_inserted, v_created_at;
END;
$$;

-- ---------------------------------------------------------------------------
-- RPC: match_resume_chunks — exact cosine top-k within one conversation
-- ---------------------------------------------------------------------------
--
-- Filters by BOTH user_id and thread_id before anything is scored. Summary is
-- excluded here, at retrieval time — the one chunking-level change the eval
-- confirmed — and no other section is filtered. Ties break by chunk_index,
-- the same order the eval's in-memory ranking uses, so the two agree exactly.
-- k is clamped to 0..20 so a caller cannot ask for an unbounded scan result.

CREATE OR REPLACE FUNCTION public.match_resume_chunks(
    p_user_id INTEGER,
    p_thread_id TEXT,
    p_query_embedding extensions.vector(1536),
    p_match_count INTEGER DEFAULT 3
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INTEGER,
    section TEXT,
    content TEXT,
    token_count INTEGER,
    page INTEGER,
    similarity DOUBLE PRECISION
)
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = public, extensions
AS $$
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.section,
        c.content,
        c.token_count,
        c.page,
        1 - (c.embedding <=> p_query_embedding) AS similarity
    FROM public.resume_chunks AS c
    WHERE c.user_id = p_user_id
      AND c.thread_id = p_thread_id
      AND c.section <> 'summary'
    ORDER BY c.embedding <=> p_query_embedding, c.chunk_index
    LIMIT GREATEST(LEAST(COALESCE(p_match_count, 3), 20), 0);
$$;

-- New functions are executable by PUBLIC by default. Take that away first.
REVOKE ALL ON FUNCTION public.replace_resume(INTEGER, TEXT, TEXT, TEXT, INTEGER, TEXT, JSONB, JSONB)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.match_resume_chunks(INTEGER, TEXT, extensions.vector, INTEGER)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.replace_resume(INTEGER, TEXT, TEXT, TEXT, INTEGER, TEXT, JSONB, JSONB)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.match_resume_chunks(INTEGER, TEXT, extensions.vector, INTEGER)
    TO service_role;

-- PostgREST caches the schema. Without a reload the new tables and functions
-- return PGRST202/PGRST205 until the cache next refreshes.
NOTIFY pgrst, 'reload schema';
