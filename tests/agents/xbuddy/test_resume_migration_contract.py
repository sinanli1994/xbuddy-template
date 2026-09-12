"""Migration 003's contract, checked against the code that depends on it.

Text-level checks over supabase/migrations/003_resume_rag.sql. They pin the
properties the Python side relies on — parameter names, the section vocabulary,
the dimension count, the Summary exclusion, the access model — so the SQL and the
adapter cannot drift apart silently.

The migration was also *executed* on Postgres 18 + pgvector 0.8.1 (PGlite) during
development, with payloads from the real adapter; these checks are what stays in
the suite, since CI has no Postgres.

Comments are stripped first: the header's prose ("no DROP", "no vector index")
would otherwise satisfy or fail checks about what the SQL actually does.
"""

import re
from pathlib import Path

import pytest

from agents.xbuddy.resume.models import ResumeSection
from agents.xbuddy.resume.retrieval import RESUME_TOP_K
from agents.xbuddy.resume.store import (
    MAX_MATCH_COUNT,
    ChunkRecord,
    build_match_payload,
    build_replace_payload,
)
from core.llm import EMBEDDING_DIMENSIONS

MIGRATION = Path(__file__).resolve().parents[3] / "supabase" / "migrations" / "003_resume_rag.sql"


def strip_comments(sql: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


RAW = MIGRATION.read_text(encoding="utf-8")
SQL = strip_comments(RAW)
FLAT = re.sub(r"\s+", " ", SQL)


def function_body(name: str) -> str:
    match = re.search(rf"FUNCTION public\.{name}\(.*?\$\$(.*?)\$\$", SQL, re.DOTALL)
    assert match, f"{name} not found"
    return re.sub(r"\s+", " ", match.group(1))


def parameter_names(name: str) -> list[str]:
    match = re.search(rf"CREATE OR REPLACE FUNCTION public\.{name}\((.*?)\)\s*RETURNS", SQL, re.DOTALL)
    assert match, f"{name} signature not found"
    return [part.strip().split()[0] for part in match.group(1).split(",")]


# --------------------------------------------------------------------------
# Shape and idempotency
# --------------------------------------------------------------------------


def test_the_migration_follows_the_numbering():
    names = sorted(p.name for p in MIGRATION.parent.glob("*.sql"))
    assert names.index("003_resume_rag.sql") == names.index("002_final_outputs.sql") + 1


def test_pgvector_is_enabled_in_the_extensions_schema():
    assert "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;" in SQL


def test_every_create_is_idempotent():
    assert re.findall(r"CREATE TABLE (?!IF NOT EXISTS)", SQL) == []
    assert re.findall(r"CREATE INDEX (?!IF NOT EXISTS)", SQL) == []
    assert re.findall(r"CREATE FUNCTION", SQL) == []  # only CREATE OR REPLACE


def test_nothing_is_dropped_or_truncated():
    assert not re.search(r"\b(DROP|TRUNCATE)\b", SQL, re.IGNORECASE)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_the_vector_dimension_matches_the_embedding_model():
    assert f"extensions.vector({EMBEDDING_DIMENSIONS})" in SQL
    assert re.findall(r"vector\((\d+)\)", SQL) == [str(EMBEDDING_DIMENSIONS)] * len(re.findall(r"vector\(\d+\)", SQL))


def test_one_resume_per_conversation():
    assert "UNIQUE (user_id, thread_id)" in FLAT


def test_chunk_indices_are_unique_per_document():
    assert "UNIQUE (document_id, chunk_index)" in FLAT


def test_chunks_are_deleted_with_their_document():
    assert "REFERENCES public.resume_documents(id) ON DELETE CASCADE" in FLAT


def test_the_scope_index_exists():
    assert re.search(r"CREATE INDEX IF NOT EXISTS \w+ ON public\.resume_chunks \(user_id, thread_id\)", FLAT)


def test_no_approximate_vector_index():
    """Exact search over one conversation's ~10-20 chunks; an ANN index under a
    filter this selective could return fewer than k rows."""
    assert not re.search(r"\b(hnsw|ivfflat)\b", SQL, re.IGNORECASE)


def test_the_section_vocabulary_matches_the_chunker_exactly():
    """The Summary filter relies on this. A section the CHECK allowed but the
    chunker never wrote is harmless; one the chunker writes but the CHECK rejects
    would fail every upload; a differently spelled 'summary' would evade the filter."""
    match = re.search(r"section TEXT NOT NULL CHECK \(section IN \((.*?)\)\)", FLAT)
    assert match
    allowed = {value.strip().strip("'") for value in match.group(1).split(",")}
    assert allowed == {section.value for section in ResumeSection}


def test_section_is_required():
    assert "section TEXT NOT NULL" in FLAT


def test_the_raw_pdf_is_never_stored():
    assert not re.search(r"\b(bytea|pdf_bytes|file_data|raw_text)\b", SQL, re.IGNORECASE)


# --------------------------------------------------------------------------
# Access: backend only
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["resume_documents", "resume_chunks"])
def test_rls_is_enabled(table):
    assert f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY;" in SQL


def test_no_policies_are_created():
    assert not re.search(r"\bCREATE POLICY\b", SQL, re.IGNORECASE)


@pytest.mark.parametrize("table", ["resume_documents", "resume_chunks"])
def test_all_api_roles_are_revoked_before_anything_is_granted(table):
    """Including service_role. Production's default privileges grant TRUNCATE,
    REFERENCES, TRIGGER and MAINTAIN to every API role on new tables; the first
    production dry run found service_role holding them on top of the intended
    grants. Revoke-all-then-grant is what makes the final set exact."""
    revoke = f"REVOKE ALL ON TABLE public.{table} FROM anon, authenticated, service_role;"
    grant = f"GRANT SELECT, INSERT, DELETE ON TABLE public.{table} TO service_role;"
    assert revoke in SQL and grant in SQL
    assert SQL.index(revoke) < SQL.index(grant)


def test_nothing_is_ever_granted_to_browser_roles():
    grants = re.findall(r"GRANT [^;]*;", FLAT)
    assert grants
    assert not any(re.search(r"\b(anon|authenticated|PUBLIC)\b", g) for g in grants)


def test_service_role_gets_exactly_what_the_adapter_uses():
    assert "GRANT SELECT, INSERT, DELETE ON TABLE public.resume_documents TO service_role;" in SQL
    assert "GRANT SELECT, INSERT, DELETE ON TABLE public.resume_chunks TO service_role;" in SQL
    assert not re.search(r"GRANT [^;]*\bUPDATE\b", FLAT)


@pytest.mark.parametrize("function", ["replace_resume", "match_resume_chunks"])
def test_functions_are_revoked_from_public_before_being_granted(function):
    revoke = FLAT.index(f"REVOKE ALL ON FUNCTION public.{function}(")
    grant = FLAT.index(f"GRANT EXECUTE ON FUNCTION public.{function}(")
    assert revoke < grant
    assert "FROM PUBLIC, anon, authenticated" in FLAT[revoke : revoke + 200]


@pytest.mark.parametrize("function", ["replace_resume", "match_resume_chunks"])
def test_functions_run_as_the_caller_with_a_pinned_search_path(function):
    header = re.search(rf"FUNCTION public\.{function}\(.*?AS \$\$", FLAT).group(0)
    assert "SECURITY INVOKER" in header and "SECURITY DEFINER" not in header
    assert "SET search_path = public, extensions" in header


def test_postgrest_reloads_its_schema_cache():
    assert "NOTIFY pgrst, 'reload schema';" in SQL


# --------------------------------------------------------------------------
# replace_resume
# --------------------------------------------------------------------------


def test_replace_parameters_match_the_adapter_payload():
    """PostgREST calls functions by parameter name. A key the SQL does not declare
    is a 404 in production, not a type error here."""
    payload = build_replace_payload(
        user_id=1, thread_id="t", filename="f", content_sha256="a" * 64, page_count=1,
        embedding_model="m", dimensions=4, candidate_facts=None,
        chunks=[ChunkRecord(chunk_index=0, section="experience", content="x", token_count=1, embedding=[0.1] * 4)],
    )
    assert parameter_names("replace_resume") == list(payload)


def test_replace_chunk_fields_match_the_adapter():
    body = function_body("replace_resume")
    fields = set(re.findall(r"c ->> '(\w+)'", body))
    assert fields == {"chunk_index", "section", "content", "token_count", "page", "embedding"}


def test_replace_deletes_only_this_conversation():
    body = function_body("replace_resume")
    assert "DELETE FROM public.resume_documents d WHERE d.user_id = p_user_id AND d.thread_id = p_thread_id;" in body


def test_replace_serialises_concurrent_uploads():
    assert "pg_advisory_xact_lock(" in function_body("replace_resume")


def test_replace_verifies_every_chunk_was_inserted():
    body = function_body("replace_resume")
    assert "GET DIAGNOSTICS v_inserted = ROW_COUNT;" in body
    assert "IF v_inserted <> v_expected THEN RAISE EXCEPTION" in body


def test_replace_is_plpgsql_so_it_is_one_transaction():
    header = re.search(r"FUNCTION public\.replace_resume\(.*?AS \$\$", FLAT).group(0)
    assert "LANGUAGE plpgsql" in header


# --------------------------------------------------------------------------
# match_resume_chunks
# --------------------------------------------------------------------------


def test_match_parameters_match_the_adapter_payload():
    payload = build_match_payload(user_id=1, thread_id="t", query_embedding=[0.1] * 4, dimensions=4, k=3)
    assert parameter_names("match_resume_chunks") == list(payload)


def test_match_filters_by_both_user_and_thread():
    body = function_body("match_resume_chunks")
    assert "WHERE c.user_id = p_user_id AND c.thread_id = p_thread_id" in body


def test_match_excludes_summary_and_nothing_else():
    """003 as shipped. Migration 004 replaces this function to exclude Header too;
    that file's own contract is checked in test_resume_retrieval_exclusions.py."""
    body = function_body("match_resume_chunks")
    assert f"AND c.section <> '{ResumeSection.SUMMARY.value}'" in body
    assert len(re.findall(r"c\.section\b", body.split("FROM", 1)[1])) == 1  # one section predicate


def test_match_is_exact_cosine_ordered_with_a_deterministic_tie_break():
    body = function_body("match_resume_chunks")
    assert "1 - (c.embedding <=> p_query_embedding) AS similarity" in body
    assert "ORDER BY c.embedding <=> p_query_embedding, c.chunk_index" in body


def test_match_defaults_to_the_production_k_and_clamps_to_the_adapter_limit():
    assert "p_match_count INTEGER DEFAULT 3" in FLAT
    assert RESUME_TOP_K == 3
    body = function_body("match_resume_chunks")
    assert f"LIMIT GREATEST(LEAST(COALESCE(p_match_count, 3), {MAX_MATCH_COUNT}), 0)" in body


def test_match_returns_every_field_the_adapter_reads():
    returns = re.search(r"FUNCTION public\.match_resume_chunks\(.*?RETURNS TABLE \((.*?)\) LANGUAGE", FLAT).group(1)
    columns = [part.strip().split()[0] for part in returns.split(",")]
    from agents.xbuddy.resume.store import RetrievedChunk

    assert columns == list(RetrievedChunk.model_fields)
