"""Section-aware resume chunking (Stage 1: offline, deterministic, no model).

The properties that matter for retrieval, each pinned here:
- headings are recognised conservatively, so a job title never becomes a section;
- a chunk never crosses a section, and a whole entry is never cut arbitrarily;
- only an entry over the hard maximum is split, and then with overlap and a
  continuation prefix naming the entry;
- a wrapped PDF produces the same chunks as the same text unwrapped;
- with no recognisable heading, the whole text falls back to overlapping windows.
"""

import re
from itertools import pairwise
from pathlib import Path

import pytest
from resume_pdf import text_pdf

from agents.xbuddy.resume.chunking import (
    HARD_MAX_TOKENS,
    SPLIT_OVERLAP_TOKENS,
    TARGET_MAX_TOKENS,
    WINDOW_OVERLAP_TOKENS,
    WINDOW_TOKENS,
    chunk_document,
    chunk_fixed_window,
    chunk_text,
    heading_section,
)
from agents.xbuddy.resume.extraction import extract_pdf_text
from agents.xbuddy.resume.models import ResumeSection
from agents.xbuddy.resume.tokens import count_tokens, split_to_budget

CORPUS = Path(__file__).resolve().parents[3] / "evals" / "resume_retrieval" / "corpus"
NAMES = ("backend_to_ai", "data_analyst", "teacher_to_ux")


def resume(name: str) -> str:
    return (CORPUS / f"{name}.txt").read_text(encoding="utf-8")


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Heading detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "section"),
    [
        ("EXPERIENCE", ResumeSection.EXPERIENCE),
        ("Work Experience:", ResumeSection.EXPERIENCE),
        ("Relevant Experience", ResumeSection.EXPERIENCE),
        ("— PROJECTS —", ResumeSection.PROJECTS),
        ("Open-Source Projects", ResumeSection.PROJECTS),
        ("Portfolio", ResumeSection.PROJECTS),
        ("Certifications & Training", ResumeSection.CERTIFICATIONS),
        ("Technical Skills:", ResumeSection.SKILLS),
        ("Profile:", ResumeSection.SUMMARY),
        ("## Education", ResumeSection.EDUCATION),
        ("TALKS", ResumeSection.PUBLICATIONS),
    ],
)
def test_common_headings_are_recognised(line, section):
    assert heading_section(line) is section


@pytest.mark.parametrize(
    "line",
    [
        "SENIOR BACKEND ENGINEER",  # a job title in capitals
        "ACME CORPORATION",  # a company name in capitals
        "• Experience",  # a one-word bullet
        "- Skills",
        "Achievements:",  # a label inside a job entry
        "Languages",  # a label inside Skills
        "Tools",
        "Skills: Python, SQL, Go",  # a skills line, not a heading
        "Experience with distributed systems at scale",  # a sentence
        "",
    ],
)
def test_things_that_are_not_headings_are_not(line):
    assert heading_section(line) is None


@pytest.mark.parametrize(
    ("name", "sections"),
    [
        ("backend_to_ai", ["summary", "experience", "projects", "education", "skills",
                           "certifications", "publications", "awards"]),
        ("data_analyst", ["summary", "experience", "projects", "education", "skills", "certifications"]),
        ("teacher_to_ux", ["summary", "experience", "projects", "education", "certifications", "volunteering"]),
    ],
)
def test_corpus_sections_are_found_in_order(name, sections):
    result = chunk_text(resume(name))
    assert result.strategy == "section"
    assert [s.value for s in result.sections] == sections


# --------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_chunk_fits_the_hard_maximum(name):
    for chunk in chunk_text(resume(name)).chunks:
        assert chunk.token_count <= HARD_MAX_TOKENS
        assert chunk.token_count == count_tokens(chunk.content)


def test_a_job_entry_is_kept_whole_and_apart_from_the_next_job():
    chunks = chunk_text(resume("backend_to_ai")).chunks
    northwind = [c for c in chunks if "Northwind Logistics" in c.text and c.section is ResumeSection.EXPERIENCE]
    assert len(northwind) == 1
    body = northwind[0].text
    assert "Mentor four engineers" in body and "chaired twelve blameless post-mortems" in body
    assert "Brightpath" not in body and "Lumen Health" not in body


@pytest.mark.parametrize("name", ["backend_to_ai", "data_analyst"])
def test_no_line_is_repeated_across_chunks_without_a_split(name):
    """Overlap exists only to repair a forced cut. Between whole entries it would
    only duplicate evidence and inflate Recall@K."""
    seen: dict[str, int] = {}
    for chunk in chunk_text(resume(name)).chunks:
        for line in chunk.text.splitlines():
            seen[line] = seen.get(line, 0) + 1
    assert [line for line, count in seen.items() if count > 1] == []


def test_a_chunk_never_crosses_a_section():
    text = resume("data_analyst")
    for chunk in chunk_text(text).chunks:
        for line in chunk.text.splitlines():
            assert heading_section(line) is None, f"a heading leaked into a chunk: {line!r}"


def test_small_entries_merge_up_to_the_target_by_default():
    chunks = chunk_text(resume("backend_to_ai")).chunks
    merged = next(c for c in chunks if "Lumen Health" in c.text)
    assert "Brightpath Payments" in merged.text  # both small, so they share a chunk
    assert merged.token_count <= TARGET_MAX_TOKENS


def test_a_small_entry_does_not_absorb_a_large_neighbour():
    """The minimum is a reason to merge; the maximum is a limit on it. In the corpus
    the minimum always stops merging first, so this case needs building."""
    big_bullets = "\n".join(
        f"• Delivered workstream {n} covering design, rollout, on-call hand-over and the "
        f"post-launch review for the platform team." for n in range(11)
    )
    big_entry = f"Lead Engineer — Big Co\n{big_bullets}"
    # The premise: too big to merge under the target, small enough not to be split.
    assert TARGET_MAX_TOKENS < count_tokens(big_entry) < HARD_MAX_TOKENS - 10

    chunks = chunk_text(f"EXPERIENCE\nIntern — Small Co\n• Wrote scripts.\n{big_entry}\n").chunks
    small = next(c for c in chunks if "Small Co" in c.text)
    assert "Big Co" not in small.text
    assert small.token_count <= TARGET_MAX_TOKENS
    assert all(c.token_count <= HARD_MAX_TOKENS for c in chunks)


def test_merging_can_be_disabled_for_one_entry_per_chunk():
    chunks = chunk_text(resume("backend_to_ai"), merge_below=0).chunks
    lumen = next(c for c in chunks if "Lumen Health" in c.text)
    assert "Brightpath Payments" not in lumen.text


# --------------------------------------------------------------------------
# Splitting an oversized entry
# --------------------------------------------------------------------------


def teacher_pieces():
    chunks = chunk_text(resume("teacher_to_ux")).chunks
    return [c for c in chunks if "Riverside Secondary School" in c.content]


def test_only_an_entry_over_the_hard_maximum_is_split():
    pieces = teacher_pieces()
    assert len(pieces) >= 2
    whole = squash("\n".join(line for line in resume("teacher_to_ux").splitlines()
                              if line.startswith("• ") and "Riverside" not in line))
    assert count_tokens(whole) > HARD_MAX_TOKENS - 20  # it really was oversized


def test_split_pieces_stay_inside_the_target():
    for piece in teacher_pieces():
        assert piece.token_count <= TARGET_MAX_TOKENS


def test_a_continuation_piece_names_its_entry():
    first, second, *_ = teacher_pieces()
    assert first.content.startswith("[Experience] Mathematics and Computer Science Teacher")
    assert second.content.startswith("[Experience — Mathematics and Computer Science Teacher, Riverside")


def test_consecutive_pieces_overlap_by_roughly_the_budget():
    first, second, *_ = teacher_pieces()
    overlap = next(
        tail for n in range(len(second.text), 0, -1)
        if first.text.endswith(tail := second.text[:n])
    )
    assert SPLIT_OVERLAP_TOKENS <= count_tokens(overlap) <= 2 * SPLIT_OVERLAP_TOKENS


def test_a_continuation_starts_on_a_whole_bullet():
    _, second, *_ = teacher_pieces()
    assert second.text.startswith("• ")


def test_split_to_budget_leaves_short_text_alone():
    assert split_to_budget("short text", max_tokens=50, overlap_tokens=10) == ["short text"]


def test_split_to_budget_rejects_an_overlap_as_large_as_the_budget():
    with pytest.raises(ValueError):
        split_to_budget("x " * 100, max_tokens=20, overlap_tokens=20)


def test_split_to_budget_survives_one_enormous_word():
    pieces = split_to_budget("see " + "a" * 3000 + " end", max_tokens=100, overlap_tokens=10)
    assert all(count_tokens(p) <= 100 for p in pieces)


# --------------------------------------------------------------------------
# Wrapped PDFs
# --------------------------------------------------------------------------


WRAPPED = """EXPERIENCE
Senior Engineer — Alpha Corp
• Designed an event pipeline on Kafka and PostgreSQL that cut end-to-end tracking
latency from 9 minutes to 45 seconds.
• Mentored four engineers
and ran the weekly architecture review.
Staff Engineer — Beta Inc
• Led the platform team.
"""


def test_a_wrapped_bullet_is_joined_back_onto_its_bullet():
    chunks = chunk_text(WRAPPED, merge_below=0).chunks
    alpha = next(c for c in chunks if "Alpha Corp" in c.text)
    assert "• Designed an event pipeline on Kafka and PostgreSQL that cut end-to-end tracking " \
           "latency from 9 minutes to 45 seconds." in alpha.text
    assert "• Mentored four engineers and ran the weekly architecture review." in alpha.text


def test_the_next_real_job_title_still_starts_a_new_entry():
    chunks = chunk_text(WRAPPED, merge_below=0).chunks
    alpha = next(c for c in chunks if "Alpha Corp" in c.text)
    beta = next(c for c in chunks if "Beta Inc" in c.text)
    assert alpha is not beta
    assert "Beta Inc" not in alpha.text


@pytest.mark.parametrize("name", NAMES)
def test_a_wrapped_multi_page_pdf_chunks_exactly_like_its_text(name):
    """The strongest check on entry splitting: pypdf returns every long bullet as
    several physical lines, and the chunks must still come out the same."""
    text = resume(name)
    from_pdf = chunk_document(extract_pdf_text(text_pdf(text)))
    from_text = chunk_text(text)
    assert [squash(c.text) for c in from_pdf.chunks] == [squash(c.text) for c in from_text.chunks]
    assert [c.section for c in from_pdf.chunks] == [c.section for c in from_text.chunks]


# --------------------------------------------------------------------------
# Prefixes
# --------------------------------------------------------------------------


def test_content_is_the_prefix_then_the_text():
    for chunk in chunk_text(resume("data_analyst")).chunks:
        assert chunk.content.endswith(chunk.text)
        assert chunk.content.startswith("[")


def test_differently_worded_headings_get_the_same_label():
    labels = {
        name: {c.content.split("]")[0] for c in chunk_text(resume(name)).chunks
               if c.section is ResumeSection.EXPERIENCE}
        for name in NAMES
    }
    # "EXPERIENCE", "Work Experience:", "Relevant Experience" all read as Experience.
    assert all(any(label.startswith("[Experience") for label in found) for found in labels.values())


def test_an_informative_heading_is_kept_in_the_prefix():
    contents = [c.content for c in chunk_text(resume("backend_to_ai")).chunks]
    assert any(c.startswith("[Publications (Talks)]") for c in contents)


def test_a_redundant_heading_is_not_repeated():
    contents = [c.content for c in chunk_text(resume("data_analyst")).chunks]
    assert not any("Work Experience" in c.split("]")[0] for c in contents)


# --------------------------------------------------------------------------
# Fallback and determinism
# --------------------------------------------------------------------------


NO_HEADINGS = " ".join(
    f"Sentence number {n} describes some work that was done at a company." for n in range(80)
)


def test_text_with_no_recognisable_heading_falls_back_to_windows():
    result = chunk_text(NO_HEADINGS)
    assert result.strategy == "window"
    assert result.sections == []
    assert len(result.chunks) >= 3
    assert all(c.token_count <= WINDOW_TOKENS for c in result.chunks)
    assert all(c.section is ResumeSection.OTHER and c.heading is None for c in result.chunks)


def test_fallback_windows_overlap():
    chunks = chunk_fixed_window(NO_HEADINGS).chunks
    for previous, current in pairwise(chunks):
        head = " ".join(current.text.split()[:6])
        assert head in previous.text
        assert count_tokens(current.text) > WINDOW_OVERLAP_TOKENS


@pytest.mark.parametrize("name", NAMES)
def test_chunking_is_deterministic(name):
    assert chunk_text(resume(name)) == chunk_text(resume(name))


def test_chunk_indices_are_sequential():
    chunks = chunk_text(resume("backend_to_ai")).chunks
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_the_header_block_is_its_own_chunk():
    first = chunk_text(resume("backend_to_ai")).chunks[0]
    assert first.section is ResumeSection.HEADER
    assert first.text.startswith("JORDAN AVERY")
