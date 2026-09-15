-- ============================================================================
-- Phase 8 — Resume RAG: exclude Header from retrieval
--
-- 003 excluded Summary at retrieval time, the one chunking-level change the
-- labelled eval measured as a gain. The Stage 4 manual run then showed the other
-- section that should never be evidence: Header — the block above the first
-- heading, which is name, email, phone and links. It ranked second for a Skill
-- Assessment query, purely on a tagline ("Senior Backend Engineer moving into AI
-- engineering"), and sent the person's contact details to the model in place of
-- evidence.
--
-- This migration replaces match_resume_chunks so the exclusion list is
-- ('summary', 'header'). Nothing else changes: the same signature, the same
-- scoping by user_id AND thread_id, the same tie-break and clamp. Because the
-- signature is unchanged, CREATE OR REPLACE keeps the existing grants; they are
-- re-stated below anyway so a fresh project applying 003 then 004 lands in the
-- same place as an existing one.
--
-- Skills written inside a Summary are not lost to the Summary exclusion: they are
-- promoted to a Skills chunk at chunk time (resume/chunking.py), before anything
-- is embedded or stored, so they are retrievable as Skills.
--
-- Safe to re-run. No DROP, no data change, no index change.
-- ============================================================================

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
      AND c.section NOT IN ('summary', 'header')
    ORDER BY c.embedding <=> p_query_embedding, c.chunk_index
    LIMIT GREATEST(LEAST(COALESCE(p_match_count, 3), 20), 0);
$$;

-- Unchanged from 003, re-stated so this file is complete on its own.
REVOKE ALL ON FUNCTION public.match_resume_chunks(INTEGER, TEXT, extensions.vector, INTEGER)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.match_resume_chunks(INTEGER, TEXT, extensions.vector, INTEGER)
    TO service_role;

-- PostgREST caches the schema; without this the replaced function can keep
-- answering from the cached definition.
NOTIFY pgrst, 'reload schema';
