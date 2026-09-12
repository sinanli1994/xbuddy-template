"""Data shapes for an uploaded resume: the extracted document and its chunks.

Deliberately free of LangGraph, model, and database imports. Stage 1 of Resume RAG
is offline — extraction and chunking are pure functions over bytes and text — and
keeping the shapes here dependency-free is what lets the retrieval eval and the
unit tests run without a key, a network, or a checkpointer.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResumeSection(StrEnum):
    """Canonical resume sections.

    Headings are normalised onto these so "Work History", "Professional
    Experience" and "EXPERIENCE" all label their chunks the same way. That matters
    twice: the section becomes part of the embedded text, and a later stage can
    filter or weight by it without caring how a particular resume phrased it.
    """

    HEADER = "header"  # everything before the first recognised heading
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    EDUCATION = "education"
    SKILLS = "skills"
    CERTIFICATIONS = "certifications"
    PUBLICATIONS = "publications"
    AWARDS = "awards"
    VOLUNTEERING = "volunteering"
    # A recognised heading with no better home (Interests, Languages), and the
    # label for every window when no heading was recognised at all.
    OTHER = "other"


# How a section is written in a chunk's contextual prefix.
SECTION_LABELS: dict[ResumeSection, str] = {
    ResumeSection.HEADER: "Header",
    ResumeSection.SUMMARY: "Summary",
    ResumeSection.EXPERIENCE: "Experience",
    ResumeSection.PROJECTS: "Projects",
    ResumeSection.EDUCATION: "Education",
    ResumeSection.SKILLS: "Skills",
    ResumeSection.CERTIFICATIONS: "Certifications",
    ResumeSection.PUBLICATIONS: "Publications",
    ResumeSection.AWARDS: "Awards",
    ResumeSection.VOLUNTEERING: "Volunteering",
    ResumeSection.OTHER: "Other",
}


# Sections retrieval never returns, enforced in the match RPC and re-checked in the
# adapter. Summary is excluded because the Stage 2 eval measured it as the one
# chunking-level change worth making; Header because it is contact details — name,
# email, phone, links — which rank on a tagline and are never evidence of a skill.
# Explicit skill lists written inside a Summary are promoted to Skills at chunk
# time (chunking.promote_summary_skills), so excluding Summary does not bury them.
NON_RETRIEVABLE_SECTIONS = frozenset({ResumeSection.SUMMARY, ResumeSection.HEADER})


ChunkStrategy = Literal["section", "window"]


class ExtractedDocument(BaseModel):
    """Normalised text pulled out of one PDF.

    `text` is the page texts joined with a single newline — not a blank line — so a
    job entry that runs across a page break is not split into two entries by the
    join itself.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    pages: list[str]
    page_count: int
    char_count: int
    # SHA-256 of the uploaded bytes. Identifies the file, not its text: two exports
    # of the same resume differ here and that is correct.
    content_sha256: str


class ResumeChunk(BaseModel):
    """One retrievable unit of a resume."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    section: ResumeSection
    # The heading as the resume wrote it ("WORK EXPERIENCE"), or None for the header
    # block and for fallback windows.
    heading: str | None
    # The chunk's own text, without any prefix. What a user would recognise.
    text: str
    # What gets embedded: `text` behind a short contextual prefix such as
    # "[Experience]". Kept separate from `text` so evidence can be shown without it.
    content: str
    token_count: int = Field(ge=1)  # of `content`, which is what the model sees
    page: int | None = Field(default=None, ge=1)  # 1-based page the chunk starts on
    strategy: ChunkStrategy


class ChunkedResume(BaseModel):
    """The chunks, plus how they were produced."""

    model_config = ConfigDict(frozen=True)

    chunks: list[ResumeChunk]
    strategy: ChunkStrategy
    # Recognised sections in document order, each once. Empty under "window".
    sections: list[ResumeSection]
    total_tokens: int


class ResumeExtractionError(Exception):
    """A PDF that cannot become resume text.

    Carries a stable `code` for the upload endpoint to map onto a 4xx, and a
    `message` written for the person who uploaded the file. Upload is the one place
    Resume RAG must fail loudly: a resume that silently indexes as nothing would
    look attached and contribute nothing.
    """

    def __init__(
        self,
        code: Literal["not_pdf", "unreadable", "encrypted", "too_many_pages", "no_text"],
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
