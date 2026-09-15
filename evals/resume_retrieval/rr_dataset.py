"""Labelled retrieval queries over three synthetic resumes.

Each query names the resume it searches and the **evidence** a good retriever
should surface. Evidence is labelled by *content anchor* — a short verbatim phrase
from the resume — not by chunk id. A retrieved chunk is relevant to an evidence
item if it contains that anchor. Chunk ids change every time the chunker does;
anchors don't, so the same labels score every chunking strategy fairly, and a
strategy that cuts an anchor in half pays for it as a miss.

A query with two evidence items (two jobs that both reduced latency) needs both to
reach full recall, which is why Recall@1 can be 0.5 for it.

The queries are phrased the way JobBuddy will actually ask during Skill Assessment
and Background — "evidence of X" — and deliberately include paraphrases that share
no words with their evidence. Whether each query is `lexical` or `semantic` is
computed from term overlap, not hand-assigned, so the split cannot be tuned to
flatter one retriever.

Synthetic people, synthetic employers; no real personal data.
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

RESUMES = ("backend_to_ai", "data_analyst", "teacher_to_ux")


@dataclass(frozen=True)
class LabeledQuery:
    id: str
    resume: str
    query: str
    evidence: tuple[str, ...]
    # The section the evidence lives in — for reading results, never used in scoring.
    expect_section: str


QUERIES: tuple[LabeledQuery, ...] = (
    # ---------------------------------------------------------- backend_to_ai --
    LabeledQuery("b1", "backend_to_ai", "What is their current job title and employer?",
                 ("Northwind Logistics",), "experience"),
    LabeledQuery("b2", "backend_to_ai", "Evidence of deploying an AI application to production",
                 ("deployed the backend to Fly.io",), "projects"),
    LabeledQuery("b3", "backend_to_ai", "Experience building retrieval-augmented generation (RAG) systems",
                 ("retrieval-augmented question answering", "retrieval over 12,000 past tickets"), "projects"),
    LabeledQuery("b4", "backend_to_ai", "Mentoring or growing other engineers",
                 ("Mentor four engineers",), "experience"),
    LabeledQuery("b5", "backend_to_ai", "Improving system latency and performance",
                 ("cut end-to-end tracking latency", "p99 latency from 800 ms to 210 ms"), "experience"),
    LabeledQuery("b6", "backend_to_ai", "How they evaluate the quality of LLM outputs",
                 ("offline evaluation for the assistant", "regression eval suite"), "experience"),
    LabeledQuery("b7", "backend_to_ai", "Healthcare or regulated patient-data experience",
                 ("HL7 and FHIR integration services",), "experience"),
    LabeledQuery("b8", "backend_to_ai", "University degree",
                 ("Computer Engineering — University of Waterloo",), "education"),
    LabeledQuery("b9", "backend_to_ai", "Public speaking or conference talks",
                 ("Toronto Python Meetup",), "publications"),
    # Stage 2 additions: paraphrases sharing no stemmed term with their evidence.
    LabeledQuery("b10", "backend_to_ai", "Reducing infrastructure costs",
                 ("Cut cloud spend by 28%",), "experience"),
    LabeledQuery("b11", "backend_to_ai", "Experience as an open-source maintainer",
                 ("transactional outbox pattern",), "projects"),
    # ----------------------------------------------------------- data_analyst --
    LabeledQuery("d1", "data_analyst", "What is their current role and employer?",
                 ("Data Analyst, Maple Retail Group",), "experience"),
    LabeledQuery("d2", "data_analyst", "Building dashboards for business stakeholders",
                 ("25 Tableau dashboards",), "experience"),
    LabeledQuery("d3", "data_analyst", "Experimentation and A/B testing",
                 ("Designed and analysed A/B tests",), "experience"),
    LabeledQuery("d4", "data_analyst", "Data modelling and analytics engineering tools",
                 ("in dbt on Snowflake",), "experience"),
    LabeledQuery("d5", "data_analyst", "Machine learning experience",
                 ("gradient-boosted churn model in scikit-learn",), "projects"),
    LabeledQuery("d6", "data_analyst", "Automating recurring reporting work",
                 ("Automated the monthly store performance report",), "experience"),
    LabeledQuery("d7", "data_analyst", "Healthcare industry experience",
                 ("patient-survey data from 11 clinics",), "experience"),
    LabeledQuery("d8", "data_analyst", "Degree and field of study",
                 ("Bachelor of Science in Statistics",), "education"),
    LabeledQuery("d9", "data_analyst", "Communicating insights to senior leadership",
                 ("Present findings to directors",), "experience"),
    LabeledQuery("d10", "data_analyst", "Fixing messy, inconsistent records",
                 ("Cleaned and reconciled",), "experience"),
    # ---------------------------------------------------------- teacher_to_ux --
    LabeledQuery("t1", "teacher_to_ux", "Conducting user research and interviews",
                 ("interviewing 30 students and parents", "I interviewed five parents"), "experience"),
    LabeledQuery("t2", "teacher_to_ux", "Usability testing experience",
                 ("moderated usability tests with 12 participants",), "experience"),
    LabeledQuery("t3", "teacher_to_ux", "Accessibility and inclusive design work",
                 ("screen-reader-friendly quizzes", "passed WCAG 2.1 AA checks"), "experience"),
    LabeledQuery("t4", "teacher_to_ux", "Organising content and information architecture",
                 ("card-sorting sessions",), "experience"),
    LabeledQuery("t5", "teacher_to_ux", "Measurable business impact of a design change",
                 ("online bookings rose 22%",), "experience"),
    LabeledQuery("t6", "teacher_to_ux", "Prototyping tools and practice",
                 ("interactive prototypes in Figma",), "experience"),
    LabeledQuery("t7", "teacher_to_ux", "Leading and managing a team",
                 ("Led a team of four teachers",), "experience"),
    LabeledQuery("t8", "teacher_to_ux", "Explaining complex changes to non-expert audiences",
                 ("presenting curriculum changes in plain language",), "experience"),
    LabeledQuery("t9", "teacher_to_ux", "UX design certification",
                 ("Google UX Design Professional Certificate",), "certifications"),
    LabeledQuery("t10", "teacher_to_ux", "Starting an extracurricular group that attracted many participants",
                 ("growing it from 6 to 45 members",), "experience"),
    LabeledQuery("t11", "teacher_to_ux", "Coaching and developing junior colleagues",
                 ("Mentored eleven student teachers",), "experience"),
)


@cache
def load_resume(name: str) -> str:
    return (CORPUS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def normalize_for_match(text: str) -> str:
    """Case- and whitespace-insensitive form used to find anchors in chunks."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def contains_anchor(chunk_text: str, anchor: str) -> bool:
    return normalize_for_match(anchor) in normalize_for_match(chunk_text)


def evidence_line(resume: str, anchor: str) -> str:
    """The resume line an anchor sits on — what `lexical`/`semantic` is judged against."""
    for line in load_resume(resume).splitlines():
        if contains_anchor(line, anchor):
            return line
    raise KeyError(f"anchor not in {resume}: {anchor!r}")
