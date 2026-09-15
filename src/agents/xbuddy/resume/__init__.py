"""Resume RAG for JobBuddy.

    extraction  PDF -> normalised text, or an explicit refusal (no OCR)
    chunking    section-aware chunks with a contextual prefix
    embeddings  cache keys, cached batching, cosine ranking
    store       Supabase HTTPS + two RPCs, scoped to (user_id, thread_id); raises
    ingestion   upload path: extract, chunk, embed, persist — fails loudly
    retrieval   interactive path: top-3 evidence or nothing — never raises

Retrieval design, fixed by the labelled eval in evals/resume_retrieval/:
section-aware chunks, text-embedding-3-small at 1536 dimensions, exact cosine,
Summary excluded at retrieval time, top-3.

Resume RAG is integrated into the JobBuddy upload and LangGraph workflow, providing
unconfirmed Background candidates and retrieved evidence for grounded Skill Assessment.
"""

from .chunking import chunk_document, chunk_fixed_window, chunk_pages, chunk_text
from .extraction import extract_pdf_text, normalize_text
from .ingestion import IndexedResume, ResumeIndexingError, index_resume
from .models import (
    ChunkedResume,
    ExtractedDocument,
    ResumeChunk,
    ResumeExtractionError,
    ResumeSection,
)
from .retrieval import RESUME_TOP_K, retrieve_resume_evidence
from .store import ResumeStatus, ResumeStore, ResumeStoreError, RetrievedChunk
from .tokens import count_tokens

__all__ = [
    "RESUME_TOP_K",
    "ChunkedResume",
    "ExtractedDocument",
    "IndexedResume",
    "ResumeChunk",
    "ResumeExtractionError",
    "ResumeIndexingError",
    "ResumeSection",
    "ResumeStatus",
    "ResumeStore",
    "ResumeStoreError",
    "RetrievedChunk",
    "chunk_document",
    "chunk_fixed_window",
    "chunk_pages",
    "chunk_text",
    "count_tokens",
    "extract_pdf_text",
    "index_resume",
    "normalize_text",
    "retrieve_resume_evidence",
]
