"""Turn an uploaded PDF into normalised resume text, or refuse it clearly.

`pypdf` because it is already a dependency, is pure Python (nothing extra in the
slim Docker image), and is permissively licensed. PyMuPDF extracts better but is
AGPL; pdfplumber would be a new dependency for no gain on text PDFs.

Every failure raises `ResumeExtractionError` with a code the endpoint maps to a 4xx.
Nothing here returns an empty document: an upload that indexed as nothing would
look attached and silently contribute nothing to the conversation.

No OCR. An image-only PDF is reported as such rather than guessed at.
"""

import hashlib
import io
import logging
import re
import unicodedata

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .models import ExtractedDocument, ResumeExtractionError

logger = logging.getLogger(__name__)

MAX_PAGES = 10

# Fewer letters than this across the whole document means there is no text layer
# to speak of — a scanned resume typically yields nothing, or a stray page number.
# A real resume has thousands.
MIN_TEXT_LETTERS = 50

# Bullet glyphs as they come out of real PDFs. Word's bullets live in the Symbol
# and Wingdings fonts and extract as Private Use Area code points (U+F0B7 and
# friends), which NFKC leaves alone — so they are mapped explicitly. Without this a
# Word-exported resume has no recognisable bullets and entry splitting degrades.
_BULLET_GLYPHS = "\u2022\u25cf\u25aa\u25e6\u2023\u2043\u00b7\uf0b7\uf0a7\uf076\uf0d8\uf0fc"
_BULLET_RE = re.compile(f"[{_BULLET_GLYPHS}]")

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")
_BLANK_RUNS = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Canonical form for resume text, whether it came from a PDF or a string.

    - NFKC, which unfolds the ligatures LaTeX PDFs are full of ("ﬁ" -> "fi") and
      would otherwise split words for both tokenizers and keyword search.
    - Every bullet glyph becomes "•".
    - Runs of layout spaces collapse to one; trailing whitespace goes.
    - At most one blank line in a row, since a blank line is an entry boundary and
      three of them mean nothing more than one does.

    Idempotent, so the chunker can apply it to text that is already normalised.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _BULLET_RE.sub("•", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def extract_pdf_text(data: bytes, *, max_pages: int = MAX_PAGES) -> ExtractedDocument:
    """Extract normalised text from PDF bytes.

    Raises `ResumeExtractionError` with one of:
      not_pdf         — the bytes are not a PDF at all
      unreadable      — a PDF pypdf cannot parse or extract
      encrypted       — needs a password to open
      too_many_pages  — longer than any resume; refused before extraction
      no_text         — no text layer, which almost always means a scan
    """
    # The spec allows the header anywhere in the first 1024 bytes.
    if b"%PDF-" not in data[:1024]:
        raise ResumeExtractionError("not_pdf", "That file isn't a PDF. Please upload your resume as a PDF.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, ValueError, OSError) as exc:
        logger.info("resume: unreadable PDF (%s)", type(exc).__name__)
        raise ResumeExtractionError(
            "unreadable", "That PDF couldn't be read. It may be damaged — try exporting it again."
        ) from exc

    if reader.is_encrypted:
        # Plenty of exported resumes are "encrypted" only to restrict printing or
        # copying, with an empty user password. Those open fine and should be
        # accepted; only a PDF that genuinely needs a password is refused.
        try:
            opened = reader.decrypt("")
        except Exception as exc:
            raise ResumeExtractionError(
                "encrypted", "That PDF is password-protected. Please upload a copy without a password."
            ) from exc
        if not opened:
            raise ResumeExtractionError(
                "encrypted", "That PDF is password-protected. Please upload a copy without a password."
            )

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise ResumeExtractionError(
            "unreadable", "That PDF couldn't be read. It may be damaged — try exporting it again."
        ) from exc

    if page_count > max_pages:
        raise ResumeExtractionError(
            "too_many_pages",
            f"That PDF has {page_count} pages. Resumes up to {max_pages} pages are supported.",
        )

    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            pages.append(normalize_text(page.extract_text() or ""))
        except Exception as exc:
            logger.info("resume: extraction failed on page %d (%s)", number, type(exc).__name__)
            raise ResumeExtractionError(
                "unreadable", "Part of that PDF couldn't be read. Try exporting it again."
            ) from exc

    # A single newline, not a blank line: a page break is not an entry boundary.
    text = normalize_text("\n".join(pages))

    if _letters(text) < MIN_TEXT_LETTERS:
        raise ResumeExtractionError(
            "no_text",
            "We couldn't find any text in that PDF — it looks like a scanned image. "
            "Scanned resumes aren't supported yet; please upload a text-based PDF.",
        )

    return ExtractedDocument(
        text=text,
        pages=pages,
        page_count=page_count,
        char_count=len(text),
        content_sha256=hashlib.sha256(data).hexdigest(),
    )
