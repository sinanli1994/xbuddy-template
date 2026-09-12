"""Section-aware resume chunking, with a fixed-window fallback.

A resume already comes divided into meaning: sections under headings, and entries
— one job, one project, one degree — inside them. Fixed-size windows ignore that
and cut through the middle of an entry or blend two jobs into one chunk, which is
exactly the evidence a query about one of them needs kept apart. So:

1. Headings are recognised from a closed vocabulary, matched against the whole
   line. Conservative on purpose: a job title in capitals ("SENIOR BACKEND
   ENGINEER") or a company name must never be mistaken for a section.
2. Inside a section, an entry starts after a blank line, or where a non-bullet
   line follows a bullet — the title line of the next job. The second rule is the
   one that matters in practice: PDF extraction routinely drops blank lines.
3. Entries are packed into chunks without crossing a section. Small neighbours
   merge until the chunk reaches the target; anything that fits the hard maximum
   stays whole, because an entry is not an arbitrary cut point.
4. Only an entry larger than the hard maximum is split, with ~40 tokens of overlap
   so a fact on the cut survives in one piece. Continuation pieces repeat the
   entry's title line, so the second half of a job still says which job it is.
5. If no heading is recognised at all, the whole text falls back to overlapping
   token windows.

Every chunk's embedded `content` carries a short prefix naming its section. The
same resume phrased with "Work History" or "EXPERIENCE" produces the same label.

Pure and deterministic: no model, no network, no randomness.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .extraction import normalize_text
from .models import (
    SECTION_LABELS,
    ChunkedResume,
    ChunkStrategy,
    ExtractedDocument,
    ResumeChunk,
    ResumeSection,
)
from .tokens import count_tokens, split_to_budget

TARGET_MIN_TOKENS = 120
TARGET_MAX_TOKENS = 250
HARD_MAX_TOKENS = 350
SPLIT_OVERLAP_TOKENS = 40
WINDOW_TOKENS = 250
WINDOW_OVERLAP_TOKENS = 40

# Continuation prefixes repeat at most this many words of an entry's title line.
_TITLE_WORDS = 12

# Whole-line heading vocabulary. Keys are normalised: lowercase, "&" -> "and",
# punctuation and decoration stripped. Extending this list is the intended way to
# recognise a new heading; loosening the match is not.
#
# Some plausible headings are left out on purpose, because resumes use them as
# labels *inside* an entry and a false heading would tear that entry out of its
# section: "Achievements:" under a job, "Languages" or "Tools" under Skills, a bare
# "Training", "Overview" or "About" inside a project.
_HEADINGS: dict[str, ResumeSection] = {
    **dict.fromkeys(
        [
            "summary", "professional summary", "career summary", "executive summary",
            "profile", "professional profile", "about me",
            "objective", "career objective", "personal statement",
        ],
        ResumeSection.SUMMARY,
    ),
    **dict.fromkeys(
        [
            "experience", "work experience", "professional experience",
            "relevant experience", "industry experience", "employment",
            "employment history", "work history", "career history",
            "experience and employment",
        ],
        ResumeSection.EXPERIENCE,
    ),
    **dict.fromkeys(
        [
            "projects", "personal projects", "selected projects", "side projects",
            "key projects", "technical projects", "academic projects", "portfolio",
            "open source", "open source projects", "open source contributions",
        ],
        ResumeSection.PROJECTS,
    ),
    **dict.fromkeys(
        [
            "education", "education and training", "academic background",
            "qualifications", "academic qualifications", "education and qualifications",
        ],
        ResumeSection.EDUCATION,
    ),
    **dict.fromkeys(
        [
            "skills", "technical skills", "core skills", "key skills", "skills and tools",
            "tools and technologies", "technologies", "tech stack", "technical stack",
            "core competencies", "competencies", "technical proficiencies",
            "skills and technologies", "skills and competencies",
        ],
        ResumeSection.SKILLS,
    ),
    **dict.fromkeys(
        [
            "certifications", "certificates", "certification",
            "licenses and certifications", "certifications and training",
            "certifications and licenses", "courses and certifications",
        ],
        ResumeSection.CERTIFICATIONS,
    ),
    **dict.fromkeys(
        ["publications", "papers", "talks", "publications and talks", "speaking"],
        ResumeSection.PUBLICATIONS,
    ),
    **dict.fromkeys(
        ["awards", "honors", "honours", "awards and honors", "awards and honours"],
        ResumeSection.AWARDS,
    ),
    **dict.fromkeys(
        [
            "volunteering", "volunteer experience", "volunteer work",
            "community involvement", "leadership and volunteering",
        ],
        ResumeSection.VOLUNTEERING,
    ),
    **dict.fromkeys(
        [
            "interests", "hobbies", "additional information",
            "activities", "references", "extracurricular activities",
        ],
        ResumeSection.OTHER,
    ),
}

# An explicit skill list written inside a Summary: a short label, a colon, then
# three or more short comma-separated items — "Tech Stack: Python, Go, SQL", or a
# category of the writer's own ("Backend & Web: Python, FastAPI, Next.js").
# Deliberately structural rather than a vocabulary of labels, because resumes name
# their own categories, and deliberately narrow: prose must never be relabelled as
# skills. A sentence fails on item length or on a mid-line full stop, so
# "Specialties: building payment platforms that scale to millions of events" (one
# long item) and "I led three teams, shipped four products, and hired six people."
# (long items, trailing stop) are both left in the Summary.
_SKILL_LABEL_MAX_WORDS = 5
_SKILL_LABEL_MAX_CHARS = 40
_SKILL_MIN_ITEMS = 3
_SKILL_ITEM_MAX_WORDS = 5
# A full stop that ends a sentence, not the one inside "Fly.io" or "Next.js".
_SENTENCE_STOP = re.compile(r"\.(?:\s|$)")

_BULLET = re.compile(r"^(?:•|-|–|—|\*|o|>)\s+")
_HEADING_DECORATION = re.compile(r"^[\s#=*_\-–—:|•.]+|[\s#=*_\-–—:|•.]+$")
_TRAILING_DECORATION = re.compile(r"[#=*_\-–—|•]\s*$")


def heading_section(line: str) -> ResumeSection | None:
    """The section a line announces, or None if it is not a recognised heading."""
    if _is_bullet(line) and not _TRAILING_DECORATION.search(line):
        # "• Experience" is a list item that happens to be one word, never a heading.
        # "— PROJECTS —" is decorated at both ends, which a bullet never is.
        return None
    key = _HEADING_DECORATION.sub("", line).lower().replace("&", " and ")
    key = re.sub(r"[^a-z ]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    if not key or len(key.split()) > 5:
        return None
    return _HEADINGS.get(key)


def _is_bullet(line: str) -> bool:
    return bool(_BULLET.match(line))


def skill_list_line(line: str) -> bool:
    """Whether a line is an explicit labelled list of skills.

    Used only inside a Summary (see `promote_summary_skills`). Conservative by
    design: it answers "is this a label followed by a list of short items", and
    nothing about meaning. Nothing is invented, reworded, or inferred — a line
    either is such a list or is left exactly where it was.
    """
    text = _BULLET.sub("", line).strip()
    label, separator, payload = text.partition(":")
    if not separator:
        return False

    label, payload = label.strip(), payload.strip()
    if (
        not label
        or not label[:1].isalpha()
        or "," in label
        or len(label) > _SKILL_LABEL_MAX_CHARS
        or len(label.split()) > _SKILL_LABEL_MAX_WORDS
    ):
        return False
    return _short_item_list(payload, minimum=_SKILL_MIN_ITEMS)


def _short_item_list(payload: str, *, minimum: int) -> bool:
    """Whether `payload` is a comma-separated list of at least `minimum` short items.

    A trailing comma is ignored: PDF extraction wraps a long line, so a skill list
    routinely ends mid-list and continues on the next line.
    """
    items = [item.strip() for item in payload.rstrip(",").split(",")]
    if len(items) < minimum or not all(items):
        return False
    return all(
        len(item.split()) <= _SKILL_ITEM_MAX_WORDS and not _SENTENCE_STOP.search(item)
        for item in items
    )


def _wrapped_skill_list(line: str) -> bool:
    """Whether a line is the wrapped remainder of the skill list above it.

    Only ever consulted while the line above is an unfinished skill list — one
    that ended without closing a sentence. PDF extraction wraps at a fixed width,
    so a list breaks after a comma ("… Bearer Authentication," / "CORS, Rate
    Limiting") or inside an item ("… SQL, Data" / "Structuring"); both continue
    the list, and neither reads as a sentence.
    """
    text = _BULLET.sub("", line).strip()
    if not text or heading_section(text) is not None:
        return False
    return _short_item_list(text, minimum=1)


@dataclass
class _Line:
    text: str
    page: int


@dataclass
class _Section:
    section: ResumeSection
    heading: str | None
    lines: list[_Line]


@dataclass
class _Entry:
    lines: list[_Line]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def page(self) -> int:
        return self.lines[0].page


def _lines_with_pages(pages: list[str]) -> list[_Line]:
    lines: list[_Line] = []
    for number, page_text in enumerate(pages, start=1):
        for raw in page_text.split("\n"):
            lines.append(_Line(raw.strip(), number))
    return lines


def _split_sections(lines: list[_Line]) -> list[_Section]:
    sections = [_Section(ResumeSection.HEADER, None, [])]
    for line in lines:
        section = heading_section(line.text) if line.text else None
        if section is not None:
            sections.append(_Section(section, line.text, []))
        else:
            sections[-1].lines.append(line)
    return [s for s in sections if any(line.text for line in s.lines)]


def _summary_skill_lines(lines: list["_Line"]) -> set[int]:
    """Indices of the Summary lines that are an explicit skill list.

    A labelled line starts the run; the lines it wrapped onto join it, so a list
    split by PDF line-wrapping moves in one piece instead of leaving half of it
    stranded in the Summary — where it would be both unreachable and orphaned.
    The run ends at the first line that closes a sentence or stops looking like a
    list, so prose following a skills block stays in the Summary.
    """
    found: set[int] = set()
    open_run = False
    for index, line in enumerate(lines):
        text = line.text.strip()
        if skill_list_line(text) or (open_run and _wrapped_skill_list(text)):
            found.add(index)
        else:
            open_run = False
            continue
        open_run = not text.rstrip().endswith((".", "!", "?"))
    return found


def promote_summary_skills(sections: list["_Section"]) -> list["_Section"]:
    """Move explicit skill lists out of a Summary into a Skills section of their own.

    Retrieval never returns Summary chunks, and a real resume writes its skills as
    labelled lists *inside* the summary block — so its entire skill list was
    unreachable as evidence. Promoting those lines fixes that without weakening
    the Summary exclusion, which is the only chunking-level change the eval
    measured as a gain.

    The lines are moved, not copied, so each appears in exactly one chunk, and
    their text is preserved character for character. The Summary's own heading
    rides along, so the promoted chunk's prefix reads "[Skills (Summary)]" and the
    evidence still says where it came from — attribution without widening the
    schema. Summary prose stays in the Summary, and stays excluded.
    """
    promoted: list[_Section] = []
    for section in sections:
        if section.section is not ResumeSection.SUMMARY:
            promoted.append(section)
            continue
        moved = _summary_skill_lines(section.lines)
        if not moved:
            promoted.append(section)
            continue
        skills = [line for index, line in enumerate(section.lines) if index in moved]
        prose = [line for index, line in enumerate(section.lines) if index not in moved]
        if any(line.text for line in prose):
            promoted.append(_Section(section.section, section.heading, prose))
        promoted.append(_Section(ResumeSection.SKILLS, section.heading, skills))
    return promoted


_SENTENCE_END = tuple(".!?;:")


def _title_case_ratio(text: str) -> float:
    words = [w for w in text.split() if w[:1].isalpha()]
    if not words:
        return 0.0
    return sum(1 for w in words if w[0].isupper()) / len(words)


def _continues_bullet(bullet: str, line: str) -> bool:
    """Whether `line` is the wrapped remainder of `bullet` rather than a new entry.

    Real PDFs wrap long bullets, so the second half arrives as a non-bullet line —
    which the entry rule would otherwise read as the title of the next job and cut
    the entry mid-sentence. A continuation starts lowercase, or follows a bullet
    that has not finished its sentence and does not itself read like a Title Case
    entry heading ("Backend Engineer — Brightpath Payments").

    Known miss: a wrap that lands just before a run of proper nouns ("…with
    Salesforce and\\nZendesk APIs") looks like a heading and is treated as one.
    """
    if line[:1].islower():
        return True
    if bullet.rstrip().endswith(_SENTENCE_END):
        return False
    return _title_case_ratio(line) < 0.6


def _split_entries(lines: list[_Line]) -> list[_Entry]:
    """An entry ends at a blank line, or where a non-bullet follows a bullet.

    A wrapped bullet continuation is joined back onto its bullet first, so a PDF
    that wraps lines produces the same entries as the same text unwrapped.
    """
    entries: list[_Entry] = []
    current: list[_Line] = []
    previous_was_bullet = False
    for line in lines:
        if not line.text:
            if current:
                entries.append(_Entry(current))
                current = []
            previous_was_bullet = False
            continue
        bullet = _is_bullet(line.text)
        if current and previous_was_bullet and not bullet:
            if _continues_bullet(current[-1].text, line.text):
                current[-1] = _Line(f"{current[-1].text} {line.text}", current[-1].page)
                continue
            entries.append(_Entry(current))
            current = []
        current.append(line)
        previous_was_bullet = bullet
    if current:
        entries.append(_Entry(current))
    return entries


def _heading_display(section: ResumeSection, heading: str | None) -> str | None:
    """The resume's own heading, when it says something the canonical label doesn't.

    "TALKS" filed under Publications keeps the word "talks", which both keyword
    and embedding search would otherwise lose; "Work Experience" under Experience
    adds nothing and is dropped. All-caps headings are title-cased for reading.
    """
    if not heading:
        return None
    cleaned = _HEADING_DECORATION.sub("", heading).strip()
    if cleaned.isupper():
        cleaned = cleaned.title()
    if not cleaned or SECTION_LABELS[section].lower() in cleaned.lower():
        return None
    return cleaned


def _prefix(section: ResumeSection, heading: str | None = None, title: str | None = None) -> str:
    """The contextual header embedded with a chunk: "[Section (Heading) — Title] "."""
    label = SECTION_LABELS[section]
    display = _heading_display(section, heading)
    if display:
        label = f"{label} ({display})"
    if title:
        words = title.split()
        short = " ".join(words[:_TITLE_WORDS]) + (" …" if len(words) > _TITLE_WORDS else "")
        return f"[{label} — {short}] "
    return f"[{label}] "


def _pack_section(
    section: _Section, merge_below: int = TARGET_MIN_TOKENS
) -> Iterable[tuple[str, int, str]]:
    """Yield (text, page, prefix) for one section's chunks, in order.

    `merge_below` is the target minimum: a chunk smaller than this absorbs the next
    entry if the result still fits the target maximum. Zero gives one entry per
    chunk, which the retrieval eval uses to measure what merging costs.
    """
    entries = _split_entries(section.lines)
    base = _prefix(section.section, section.heading)
    base_tokens = count_tokens(base)

    buffer: list[_Entry] = []

    def flush() -> Iterable[tuple[str, int, str]]:
        if buffer:
            yield "\n".join(e.text for e in buffer), buffer[0].page, base
            buffer.clear()

    for entry in entries:
        entry_tokens = count_tokens(entry.text) + base_tokens

        if entry_tokens > HARD_MAX_TOKENS:
            yield from flush()
            title = entry.lines[0].text
            pieces = split_to_budget(
                entry.text,
                max_tokens=TARGET_MAX_TOKENS
                - count_tokens(_prefix(section.section, section.heading, title)),
                overlap_tokens=SPLIT_OVERLAP_TOKENS,
            )
            for number, piece in enumerate(pieces):
                # The first piece already opens with the title line.
                continuation = _prefix(section.section, section.heading, title)
                yield piece, entry.page, base if number == 0 else continuation
            continue

        if buffer:
            merged = "\n".join(e.text for e in buffer) + "\n" + entry.text
            buffered_tokens = count_tokens("\n".join(e.text for e in buffer)) + base_tokens
            # Merge only while the buffer is still below the target minimum, and
            # only if the result stays inside the target maximum.
            if (
                buffered_tokens < merge_below
                and count_tokens(merged) + base_tokens <= TARGET_MAX_TOKENS
            ):
                buffer.append(entry)
                continue
            yield from flush()
        buffer.append(entry)

    yield from flush()


def _make_chunks(
    raw: Iterable[tuple[str, int | None, str, ResumeSection, str | None]], strategy: ChunkStrategy
) -> list[ResumeChunk]:
    chunks: list[ResumeChunk] = []
    for text, page, prefix, section, heading in raw:
        content = f"{prefix}{text}" if prefix else text
        chunks.append(
            ResumeChunk(
                index=len(chunks),
                section=section,
                heading=heading,
                text=text,
                content=content,
                token_count=count_tokens(content),
                page=page,
                strategy=strategy,
            )
        )
    return chunks


def chunk_fixed_window(
    text: str,
    *,
    window_tokens: int = WINDOW_TOKENS,
    overlap_tokens: int = WINDOW_OVERLAP_TOKENS,
) -> ChunkedResume:
    """Overlapping token windows over the whole text, blind to structure.

    The fallback when no heading is recognised, and the baseline the retrieval
    eval compares section-aware chunking against.
    """
    normalized = normalize_text(text)
    pieces = split_to_budget(normalized, max_tokens=window_tokens, overlap_tokens=overlap_tokens)
    chunks = _make_chunks(
        ((piece, None, "", ResumeSection.OTHER, None) for piece in pieces), "window"
    )
    return ChunkedResume(
        chunks=chunks,
        strategy="window",
        sections=[],
        total_tokens=count_tokens(normalized),
    )


def chunk_pages(pages: list[str], *, merge_below: int = TARGET_MIN_TOKENS) -> ChunkedResume:
    """Section-aware chunks for text already split into pages."""
    normalized_pages = [normalize_text(page) for page in pages]
    sections = promote_summary_skills(_split_sections(_lines_with_pages(normalized_pages)))
    recognised = [s for s in sections if s.section is not ResumeSection.HEADER]

    full_text = normalize_text("\n".join(normalized_pages))
    if not recognised:
        return chunk_fixed_window(full_text)

    raw: list[tuple[str, int | None, str, ResumeSection, str | None]] = []
    for section in sections:
        for text, page, prefix in _pack_section(section, merge_below):
            raw.append((text, page, prefix, section.section, section.heading))

    order: list[ResumeSection] = []
    for section in recognised:
        if section.section not in order:
            order.append(section.section)

    return ChunkedResume(
        chunks=_make_chunks(raw, "section"),
        strategy="section",
        sections=order,
        total_tokens=count_tokens(full_text),
    )


def chunk_text(text: str, *, merge_below: int = TARGET_MIN_TOKENS) -> ChunkedResume:
    """Section-aware chunks for plain resume text, treated as a single page."""
    return chunk_pages([text], merge_below=merge_below)


def chunk_document(document: ExtractedDocument) -> ChunkedResume:
    """Section-aware chunks for an extracted PDF, with page attribution."""
    return chunk_pages(document.pages)
