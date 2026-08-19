-- ============================================================================
-- PR 5 — JobBuddy final-output table
--
-- The table name is `final-outputs`, hyphenated, because the existing frontend
-- already reads and writes exactly that name in
-- frontend/src/components/BusinessPlanEditor.tsx and
-- frontend/src/app/business-plan/edit/page.tsx. A hyphen is not a valid bare
-- identifier, so every reference here is double-quoted. Renaming it to
-- final_outputs would be tidier SQL but would break the frontend, and PR 5 does
-- not touch the frontend.
--
-- Follows the _ultra_safe baseline's style: IF NOT EXISTS everywhere, guarded DO
-- blocks for the trigger and the realtime publication, no DROP, no destructive
-- REPLACE. Safe to re-run.
--
-- Backward compatibility with the frontend is the binding constraint. The frontend
-- upserts exactly six columns:
--     user_id, thread_id, agent_id, content, markdown_content, updated_at
-- so every backend-only column below is either nullable or defaulted. Postgres
-- ON CONFLICT DO UPDATE only assigns the columns PostgREST was given, which is
-- what makes edit detection work: a user's save changes `content` and leaves
-- `generated_content_hash` alone, so the two stop agreeing.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS "final-outputs" (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    -- Written explicitly by both the agent and the frontend as 'xbuddy'. Not part
    -- of the unique key, matching business_plans and section_states.
    agent_id TEXT NOT NULL DEFAULT 'xbuddy',
    -- The frontend writes the same Markdown into both columns and reads
    -- `markdown_content || content`, so the agent writes both identically too.
    -- TEXT rather than JSONB: the editor's convertToTiptapFormat parses Markdown,
    -- and Markdown is PR 5's canonical artifact format.
    content TEXT NOT NULL,
    markdown_content TEXT,
    -- Backend-only. 'current' means the document matches the graph's live
    -- user_data; 'stale' means source memory changed after it was generated and
    -- the content is retained but no longer authoritative. Defaulted so the
    -- frontend's six-column upsert can insert a row without knowing about it.
    status TEXT NOT NULL DEFAULT 'current' CHECK (status IN ('current', 'stale')),
    -- Backend-only. SHA-256 of the canonical Markdown the agent last generated.
    -- Nullable because the frontend never sets it: a row it created on its own has
    -- no agent-generated version to compare against.
    generated_content_hash TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- The conflict target the frontend already assumes, and the one the agent
    -- passes explicitly as on_conflict="user_id,thread_id".
    UNIQUE(user_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_final_outputs_user_thread
    ON "final-outputs"(user_id, thread_id);

-- Reuse the baseline's shared trigger function when it is already present, and
-- define it otherwise, so this migration also applies to a project that has only
-- ever had 002 run against it.
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- PostgreSQL has no CREATE TRIGGER IF NOT EXISTS, so guard on pg_trigger.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'update_final_outputs_updated_at'
    ) THEN
        CREATE TRIGGER update_final_outputs_updated_at
        BEFORE UPDATE ON "final-outputs"
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- The editor subscribes to postgres_changes UPDATE events on this table, so it
-- has to be in the realtime publication for live refresh to work.
--
-- Note: the baseline migration's equivalent block adds a table only when it is
-- *already* in the publication (`IF EXISTS ... THEN ALTER PUBLICATION ADD`), which
-- is inverted and does nothing; its EXCEPTION handler hides that. This block uses
-- IF NOT EXISTS, which is what was intended. The handler is kept for the real case
-- it covers: a project where the supabase_realtime publication does not exist.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime' AND tablename = 'final-outputs'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE "final-outputs";
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not add final-outputs to supabase_realtime: %', SQLERRM;
END $$;

-- Deliberately not here (PR 6): RLS policies. New tables in this project get RLS
-- enabled with no policies, and the agent's secret key bypasses RLS so its writes
-- succeed. The frontend reads this table directly, so those reads need policies
-- before the UI works against a non-service key. Out of scope for PR 5, and
-- documented in evals/... and the PR body rather than half-done here.
