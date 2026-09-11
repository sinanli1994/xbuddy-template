"""PDF text extraction for Resume RAG (Stage 1: offline, no model, no database).

Upload is the one place Resume RAG must fail loudly, so most of these pin a refusal:
each unusable PDF raises `ResumeExtractionError` with a stable code the endpoint will
map onto a 4xx, and a message written for the person who uploaded it. A resume that
indexed as nothing would look attached and contribute nothing.
"""

import hashlib
from pathlib import Path

import pytest
from resume_pdf import encrypted_pdf, image_only_pdf, pages_pdf, text_pdf

from agents.xbuddy.resume.chunking import chunk_document, chunk_pages
from agents.xbuddy.resume.extraction import extract_pdf_text, normalize_text
from agents.xbuddy.resume.models import ResumeExtractionError

CORPUS = Path(__file__).resolve().parents[3] / "evals" / "resume_retrieval" / "corpus"

SIMPLE = """Alex Doe
Toronto, ON
EXPERIENCE
Backend Engineer — Example Co
• Built the billing API in Python and PostgreSQL.
• Reduced checkout latency by 40% with a read-through cache.
EDUCATION
BSc Computer Science, Example University, 2018
"""


def refusal(data: bytes, **kwargs) -> ResumeExtractionError:
    with pytest.raises(ResumeExtractionError) as caught:
        extract_pdf_text(data, **kwargs)
    return caught.value


# --------------------------------------------------------------------------
# Readable PDFs
# --------------------------------------------------------------------------


def test_a_text_pdf_is_extracted():
    doc = extract_pdf_text(text_pdf(SIMPLE))
    assert doc.page_count == 1
    assert "Backend Engineer — Example Co" in doc.text
    assert "• Built the billing API in Python and PostgreSQL." in doc.text
    assert doc.char_count == len(doc.text)


def test_the_hash_identifies_the_uploaded_bytes():
    data = text_pdf(SIMPLE)
    assert extract_pdf_text(data).content_sha256 == hashlib.sha256(data).hexdigest()
    assert extract_pdf_text(text_pdf(SIMPLE + "\nAWARDS\nPrize\n")).content_sha256 != hashlib.sha256(data).hexdigest()


def test_non_ascii_resume_text_survives():
    """En dashes, bullets, accented place names — the characters resumes are full of."""
    doc = extract_pdf_text(text_pdf("Sam Okafor\nMontréal, QC | Aug 2021 – Present\n" + SIMPLE))
    assert "Montréal, QC | Aug 2021 – Present" in doc.text


def test_a_multi_page_pdf_keeps_its_pages():
    doc = extract_pdf_text(pages_pdf(["EXPERIENCE\nFirst page text here", "Second page text here\nEDUCATION\nBSc"]))
    assert doc.page_count == 2
    assert len(doc.pages) == 2
    assert "First page" in doc.pages[0] and "Second page" in doc.pages[1]


def test_pages_join_without_a_blank_line():
    """A blank line is an entry boundary; a page break must not become one."""
    doc = extract_pdf_text(
        pages_pdf(
            [
                "EXPERIENCE\nPlatform Engineer — Example Company\n• Built the deployment pipeline.",
                "• Rebuilt the monitoring and alerting for every service.",
            ]
        )
    )
    assert "\n\n" not in doc.text


def test_an_entry_running_across_a_page_break_stays_one_chunk():
    doc = extract_pdf_text(
        pages_pdf(
            [
                "EXPERIENCE\nSenior Engineer — Page One Corp\n• Led the platform team of six engineers.",
                "• Shipped the migration on schedule.\nEDUCATION\nBSc Physics, 2015",
            ]
        )
    )
    chunks = chunk_document(doc).chunks
    job = next(c for c in chunks if "Page One Corp" in c.text)
    assert "Shipped the migration" in job.text
    assert job.page == 1


@pytest.mark.parametrize("name", ["backend_to_ai", "data_analyst", "teacher_to_ux"])
def test_every_corpus_resume_extracts_from_a_wrapped_pdf(name):
    text = (CORPUS / f"{name}.txt").read_text(encoding="utf-8")
    doc = extract_pdf_text(text_pdf(text))
    assert doc.page_count >= 1
    assert text.splitlines()[0] in doc.text


def test_two_page_corpus_resumes_really_are_two_pages():
    """The long resumes exist to exercise page attribution; make sure they do."""
    for name in ("backend_to_ai", "teacher_to_ux"):
        text = (CORPUS / f"{name}.txt").read_text(encoding="utf-8")
        doc = extract_pdf_text(text_pdf(text))
        assert doc.page_count == 2, name
        assert {c.page for c in chunk_document(doc).chunks} == {1, 2}, name


# --------------------------------------------------------------------------
# Refusals — each with a stable code and a user-facing message
# --------------------------------------------------------------------------


def test_bytes_that_are_not_a_pdf_are_refused():
    error = refusal(b"Name: Alex\nExperience: lots")
    assert error.code == "not_pdf"
    assert "PDF" in error.message


def test_a_damaged_pdf_is_refused_as_unreadable():
    error = refusal(b"%PDF-1.4\nthis is not really a pdf at all\n%%EOF")
    assert error.code == "unreadable"


def test_an_image_only_pdf_is_refused_as_having_no_text():
    """No OCR: a scan is reported, never guessed at."""
    error = refusal(image_only_pdf())
    assert error.code == "no_text"
    assert "scanned" in error.message.lower()


def test_a_pdf_with_only_a_stray_word_counts_as_no_text():
    """A page number on an otherwise image-only page is not a resume."""
    assert refusal(text_pdf("Page 1")).code == "no_text"


def test_a_password_protected_pdf_is_refused():
    error = refusal(encrypted_pdf(SIMPLE, user_password="hunter2"))
    assert error.code == "encrypted"
    assert "password" in error.message.lower()


def test_a_restrictions_only_encrypted_pdf_is_accepted():
    """Many exported resumes are encrypted only to block printing or copying, with an
    empty user password. Those open for anyone and must not be refused."""
    doc = extract_pdf_text(encrypted_pdf(SIMPLE, user_password=""))
    assert "Backend Engineer — Example Co" in doc.text


def test_too_many_pages_is_refused_before_extraction():
    error = refusal(pages_pdf(["EXPERIENCE\nOne"] * 3), max_pages=2)
    assert error.code == "too_many_pages"
    assert "3 pages" in error.message


def test_refusal_messages_carry_no_exception_internals():
    for data in (b"not a pdf", b"%PDF-1.4\ngarbage", image_only_pdf()):
        message = refusal(data).message
        assert "Error" not in message and "Traceback" not in message and "pypdf" not in message


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_ligatures_are_unfolded():
    """LaTeX PDFs emit "ﬁ"; left alone it splits "profile" for every search."""
    assert normalize_text("Proﬁle: ﬂexible workﬂows") == "Profile: flexible workflows"


def test_word_symbol_font_bullets_become_bullets():
    """Word's bullets extract as Private Use Area code points NFKC leaves alone."""
    assert normalize_text("\uf0b7 Built things\n\uf0a7 Led things") == "• Built things\n• Led things"


def test_layout_spacing_collapses():
    assert normalize_text("Senior   Engineer\t\t—   Acme  \n\n\n\nEducation") == "Senior Engineer — Acme\n\nEducation"


def test_normalisation_is_idempotent():
    text = (CORPUS / "teacher_to_ux.txt").read_text(encoding="utf-8")
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_chunk_pages_normalises_its_input():
    assert chunk_pages(["EXPERIENCE\n\uf0b7 Built the  thing"]).chunks[0].text == "• Built the thing"
