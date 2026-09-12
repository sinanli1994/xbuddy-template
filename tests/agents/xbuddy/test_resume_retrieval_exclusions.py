"""What retrieval may return, and how a Summary's skills escape the exclusion.

Two changes, one purpose — the evidence a Skill Assessment sees should be skills
and work, never contact details or marketing prose:

- Header joins Summary on the exclusion list (migration 004 + an adapter filter).
  The Stage 4 live run returned the header block as evidence #2, on the strength
  of a tagline, sending name, email, phone and links to the model.
- Explicit skill lists written *inside* a Summary are promoted to a Skills chunk
  at chunk time. A real resume put its whole skill list there, where the Summary
  exclusion made it permanently unreachable.

Offline and deterministic: the ranking checks use a bag-of-words embedder, so
"a perfect-match header chunk is still excluded" is a measurement, not a claim.
"""

import math
import re
from pathlib import Path

import pytest

from agents.xbuddy.resume.chunking import chunk_text, promote_summary_skills, skill_list_line
from agents.xbuddy.resume.models import NON_RETRIEVABLE_SECTIONS, ResumeChunk, ResumeSection
from agents.xbuddy.resume.store import ResumeStore, RetrievedChunk

MIGRATIONS = Path(__file__).resolve().parents[3] / "supabase" / "migrations"
CORPUS = Path(__file__).resolve().parents[3] / "evals" / "resume_retrieval" / "corpus"
EXCLUSIONS = MIGRATIONS / "004_resume_retrieval_exclusions.sql"


def strip_comments(sql: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


SQL = strip_comments(EXCLUSIONS.read_text(encoding="utf-8"))


def function_body(sql: str, name: str) -> str:
    match = re.search(rf"FUNCTION public\.{name}\(.*?\$\$(.*?)\$\$", sql, re.DOTALL)
    assert match, f"{name} not found"
    return re.sub(r"\s+", " ", match.group(1))


# A resume that writes its skills inside the summary, as labelled category lines.
SUMMARY_SKILLS_RESUME = """JORDAN AVERY
jordan.avery@example.com | (+1) 555-0100 | Toronto, ON | github.com/javery-dev

SUMMARY
AI engineer with a Master's degree and hands-on experience shipping LLM agents.
Backed by four years embedded in automotive infotainment product development.
AI & Agent Engineering: LangGraph, LangChain, OpenAI API, Prompt Engineering
Backend & Web: Python, FastAPI, Next.js, REST API Design, SSE Streaming
Data & Persistence: PostgreSQL, Supabase, pgvector, SQL

EXPERIENCE
Senior Backend Engineer — Northwind Logistics
Mar 2020 – Present
• Designed an event pipeline on Kafka and PostgreSQL.

EDUCATION
MSc Computational Science — Laurentian University, 2025
"""


def chunks_of(text: str) -> list[ResumeChunk]:
    return chunk_text(text).chunks


def section_of(chunks: list[ResumeChunk], section: str) -> list[ResumeChunk]:
    return [c for c in chunks if c.section.value == section]


# --------------------------------------------------------------------------
# Fix 1 — the exclusion list: migration and adapter
# --------------------------------------------------------------------------


def test_the_migration_is_the_latest_and_idempotent():
    names = sorted(p.name for p in MIGRATIONS.glob("*.sql"))
    assert names[-1] == EXCLUSIONS.name
    assert re.findall(r"CREATE FUNCTION", SQL) == []  # CREATE OR REPLACE only
    assert not re.search(r"\b(DROP|TRUNCATE|DELETE)\b", SQL, re.IGNORECASE)


def test_the_rpc_excludes_exactly_the_python_exclusion_list():
    body = function_body(SQL, "match_resume_chunks")
    excluded = re.search(r"c\.section NOT IN \((.*?)\)", body)
    assert excluded, body
    assert {value.strip().strip("'") for value in excluded.group(1).split(",")} == {
        section.value for section in NON_RETRIEVABLE_SECTIONS
    }
    # One section predicate: nothing else is filtered, weighted, or re-ordered.
    assert len(re.findall(r"c\.section\b", body.split("FROM", 1)[1])) == 1


def test_scoping_and_ordering_are_carried_over_unchanged():
    """Only the exclusion list changes. Ownership semantics are not touched."""
    previous = function_body(
        strip_comments((MIGRATIONS / "003_resume_rag.sql").read_text(encoding="utf-8")),
        "match_resume_chunks",
    )
    body = function_body(SQL, "match_resume_chunks")
    for clause in (
        "WHERE c.user_id = p_user_id AND c.thread_id = p_thread_id",
        "1 - (c.embedding <=> p_query_embedding) AS similarity",
        "ORDER BY c.embedding <=> p_query_embedding, c.chunk_index",
        "LIMIT GREATEST(LEAST(COALESCE(p_match_count, 3), 20), 0)",
    ):
        assert clause in previous and clause in body


def test_the_function_stays_backend_only():
    flat = re.sub(r"\s+", " ", SQL)
    assert "SECURITY INVOKER" in flat
    assert "REVOKE ALL ON FUNCTION public.match_resume_chunks" in flat
    assert "FROM PUBLIC, anon, authenticated" in flat
    assert re.search(r"GRANT EXECUTE ON FUNCTION public\.match_resume_chunks.*?TO service_role", flat)
    assert "TO anon" not in flat and "TO authenticated" not in flat


def match_row(index: int, section: str = "experience", similarity: float = 0.5) -> dict:
    return {
        "id": f"c{index}",
        "document_id": "doc-1",
        "chunk_index": index,
        "section": section,
        "content": f"[{section.title()}] chunk {index}",
        "token_count": 20,
        "page": 1,
        "similarity": similarity,
    }


class FakeClient:
    def __init__(self, rows):
        self.rows = rows

    def rpc(self, name, payload):
        client = self

        class Result:
            def execute(self):
                return type("Response", (), {"data": client.rows})()

        return Result()


@pytest.mark.asyncio
@pytest.mark.parametrize("section", sorted(s.value for s in NON_RETRIEVABLE_SECTIONS))
async def test_an_excluded_row_that_arrives_anyway_is_dropped_and_reported(section, caplog):
    """If one arrives, the deployed function is not the migrated one."""
    rows = [match_row(1, section, 0.99), match_row(2)]
    store = ResumeStore(client=FakeClient(rows), dimensions=3)
    with caplog.at_level("ERROR"):
        chunks = await store.match(user_id=7, thread_id="t", query_embedding=[0.1] * 3, k=3)
    assert [c.section for c in chunks] == [ResumeSection.EXPERIENCE]
    assert section in caplog.text and "004" in caplog.text


@pytest.mark.asyncio
async def test_other_sections_pass_through_in_the_order_the_database_ranked_them():
    rows = [match_row(3, "skills", 0.8), match_row(1, "projects", 0.7), match_row(2, "education", 0.6)]
    store = ResumeStore(client=FakeClient(rows), dimensions=3)
    chunks = await store.match(user_id=7, thread_id="t", query_embedding=[0.1] * 3, k=3)
    assert [(c.section.value, c.chunk_index) for c in chunks] == [
        ("skills", 3), ("projects", 1), ("education", 2)
    ]


@pytest.mark.asyncio
async def test_no_resume_is_still_simply_no_evidence():
    store = ResumeStore(client=FakeClient([]), dimensions=3)
    assert await store.match(user_id=7, thread_id="t", query_embedding=[0.1] * 3, k=3) == []


def test_retrieved_chunks_can_still_carry_every_section():
    """The exclusion is a filter on what retrieval returns, not a schema change:
    a stored Header chunk is still a valid row."""
    assert RetrievedChunk.model_validate(match_row(1, "header")).section is ResumeSection.HEADER


# --------------------------------------------------------------------------
# Fix 1 — measured: ranking over the real chunker, excluded sections removed
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9+#.]+")


def bag(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for word in _WORD.findall(text.lower()):
        counts[word] = counts.get(word, 0.0) + 1.0
    return counts


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    shared = sum(a[w] * b[w] for w in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return shared / norm if norm else 0.0


def rank(query: str, chunks: list[ResumeChunk], *, exclude=NON_RETRIEVABLE_SECTIONS, k: int = 3):
    """What the RPC would return: excluded sections never enter the candidate set."""
    candidates = [c for c in chunks if c.section not in exclude]
    scored = sorted(candidates, key=lambda c: (-cosine(bag(query), bag(c.content)), c.index))
    return scored[:k]


def test_a_header_chunk_that_matches_perfectly_is_still_never_returned():
    chunks = chunks_of(SUMMARY_SKILLS_RESUME)
    header = next(c for c in chunks if c.section is ResumeSection.HEADER)
    # The query IS the header chunk, so it would rank first on any similarity.
    assert cosine(bag(header.content), bag(header.content)) == pytest.approx(1.0)
    returned = rank(header.content, chunks)
    assert header not in returned
    assert all(c.section not in NON_RETRIEVABLE_SECTIONS for c in returned)
    for personal in ("jordan.avery@example.com", "555-0100", "github.com/javery-dev"):
        assert all(personal not in c.content for c in returned)


def test_a_summary_chunk_that_matches_perfectly_is_still_never_returned():
    chunks = chunks_of(SUMMARY_SKILLS_RESUME)
    summary = next(c for c in chunks if c.section is ResumeSection.SUMMARY)
    returned = rank(summary.content, chunks)
    assert summary not in returned
    assert "four years embedded in automotive" not in " ".join(c.content for c in returned)


def test_other_sections_still_rank_normally():
    chunks = chunks_of(SUMMARY_SKILLS_RESUME)
    experience = next(c for c in chunks if c.section is ResumeSection.EXPERIENCE)
    education = next(c for c in chunks if c.section is ResumeSection.EDUCATION)
    assert rank("event pipeline on Kafka and PostgreSQL", chunks)[0] is experience
    assert rank("MSc Computational Science Laurentian University", chunks)[0] is education


def test_the_promoted_skills_are_reachable_at_all():
    """Before promotion these terms existed only in the Summary, so no query could
    reach them. The promoted chunk is now the single retrievable place they live."""
    chunks = chunks_of(SUMMARY_SKILLS_RESUME)
    top = rank("Python FastAPI LangGraph pgvector PostgreSQL", chunks)[0]
    assert top.section is ResumeSection.SKILLS
    assert "LangGraph" in top.text and "FastAPI" in top.text

    retrievable = [c for c in chunks if c.section not in NON_RETRIEVABLE_SECTIONS]
    assert [c.section.value for c in retrievable if "LangChain" in c.text] == ["skills"]


# --------------------------------------------------------------------------
# Fix 2 — promoting explicit skill lists out of a Summary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("line", [
    "Skills: Python, Go, SQL",
    "Technical Skills: Python, FastAPI, PostgreSQL, Docker",
    "Tools: Docker, Terraform, GitHub Actions",
    "Technologies: LangGraph, pgvector, Redis",
    "Tech Stack: Next.js, FastAPI, Fly.io",
    "Core Skills: routing, evaluation, prompt design",
    # The writer's own category labels, which no vocabulary could enumerate.
    "AI & Agent Engineering: LangGraph, LangChain, OpenAI API, Prompt Engineering",
    "Data & Persistence: PostgreSQL, Supabase, pgvector, SQL",
    "• Languages: Python, Go, Java",
])
def test_labelled_skill_lists_are_recognised(line):
    assert skill_list_line(line)


@pytest.mark.parametrize("line", [
    "AI engineer with a Master's degree and hands-on experience shipping LLM agents.",
    "Backed by four years embedded in automotive infotainment product development.",
    # A colon, but prose after it: items are long, or it ends a sentence.
    "Focus: building and operating retrieval systems that serve millions of requests daily.",
    "Summary: I led three teams, shipped four products, and hired six engineers.",
    "Achievements: cut latency from 9 minutes to 45 seconds, and reduced cloud spend by 28%.",
    # Too few items to be a list.
    "Skills: Python, Go",
    # A label that is really a sentence.
    "I have worked with many tools over the years including Docker, Kafka, and Redis: all of them",
    "",
    "Experience",
])
def test_prose_is_never_relabelled_as_skills(line):
    assert not skill_list_line(line)


def test_skills_inside_a_summary_become_a_retrievable_skills_chunk():
    chunks = chunks_of(SUMMARY_SKILLS_RESUME)
    skills = section_of(chunks, "skills")
    assert len(skills) == 1
    assert skills[0].section not in NON_RETRIEVABLE_SECTIONS
    for skill in ("LangGraph", "FastAPI", "pgvector", "SSE Streaming"):
        assert skill in skills[0].text


def test_the_promoted_chunk_says_it_came_from_the_summary():
    """Attribution without widening the schema: the resume's own heading rides
    along into the chunk's prefix."""
    skills = section_of(chunks_of(SUMMARY_SKILLS_RESUME), "skills")[0]
    assert skills.content.startswith("[Skills (Summary)] ")


def test_summary_prose_stays_in_the_summary_and_stays_excluded():
    summary = section_of(chunks_of(SUMMARY_SKILLS_RESUME), "summary")
    assert len(summary) == 1
    assert summary[0].section in NON_RETRIEVABLE_SECTIONS
    assert "AI engineer with a Master's degree" in summary[0].text
    assert "LangGraph" not in summary[0].text  # the skill lines left


def test_a_skill_line_is_never_duplicated_across_chunks():
    chunks = chunks_of(SUMMARY_SKILLS_RESUME)
    for line in SUMMARY_SKILLS_RESUME.splitlines():
        if skill_list_line(line):
            assert sum(line in c.text for c in chunks) == 1, line


def test_the_promoted_text_is_the_source_text_unchanged():
    chunks = chunks_of(SUMMARY_SKILLS_RESUME)
    skills = section_of(chunks, "skills")[0]
    for line in skills.text.splitlines():
        assert line in SUMMARY_SKILLS_RESUME  # nothing reworded, nothing invented


def test_a_wrapped_skill_list_moves_in_one_piece():
    """PDF extraction wraps at a fixed width: a list breaks after a comma, or
    inside an item. Both halves belong to the same skills chunk."""
    wrapped = SUMMARY_SKILLS_RESUME.replace(
        "Backend & Web: Python, FastAPI, Next.js, REST API Design, SSE Streaming",
        "Backend & Web: Python, FastAPI, Next.js, REST API Design, SSE\nStreaming, CORS, Rate\nLimiting",
    )
    chunks = chunks_of(wrapped)
    skills = section_of(chunks, "skills")[0]
    assert "Streaming, CORS, Rate" in skills.text and "Limiting" in skills.text
    assert "CORS" not in section_of(chunks, "summary")[0].text


def test_prose_after_a_skill_list_stays_in_the_summary():
    text = SUMMARY_SKILLS_RESUME.replace(
        "Data & Persistence: PostgreSQL, Supabase, pgvector, SQL",
        "Data & Persistence: PostgreSQL, Supabase, pgvector, SQL\n"
        "I am looking for a role where I can keep shipping agents end to end.",
    )
    chunks = chunks_of(text)
    assert "looking for a role" in section_of(chunks, "summary")[0].text
    assert "looking for a role" not in section_of(chunks, "skills")[0].text


def test_a_summary_without_skill_lists_is_left_alone():
    text = SUMMARY_SKILLS_RESUME.replace(
        "AI & Agent Engineering: LangGraph, LangChain, OpenAI API, Prompt Engineering\n", ""
    ).replace("Backend & Web: Python, FastAPI, Next.js, REST API Design, SSE Streaming\n", "").replace(
        "Data & Persistence: PostgreSQL, Supabase, pgvector, SQL\n", ""
    )
    chunks = chunks_of(text)
    assert section_of(chunks, "skills") == []
    assert "four years embedded" in section_of(chunks, "summary")[0].text


def test_promotion_is_a_no_op_for_sections_that_are_not_a_summary():
    """Only a Summary is rewritten. Everything else is returned untouched."""
    from agents.xbuddy.resume.chunking import _Line, _Section

    lines = [_Line("Tools: Docker, Terraform, GitHub Actions", 1)]
    for section in (ResumeSection.EXPERIENCE, ResumeSection.SKILLS, ResumeSection.HEADER):
        original = [_Section(section, "HEADING", lines)]
        assert promote_summary_skills(original) == original


@pytest.mark.parametrize("name", ["backend_to_ai", "data_analyst", "teacher_to_ux"])
def test_the_eval_corpus_is_unaffected(name):
    """Every corpus resume writes its skills under a heading of their own, so
    promotion never fires — the Stage 2 measurements still describe these chunks."""
    chunks = chunks_of((CORPUS / f"{name}.txt").read_text(encoding="utf-8"))
    assert all("(Summary)" not in c.content for c in chunks)
