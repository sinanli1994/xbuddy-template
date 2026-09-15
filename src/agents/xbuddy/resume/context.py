"""Resume context for the conversation: resolve it, cache it, render it.

Two sections use a resume, each in the way that fits it:

- Background proposes **candidate facts** — the four Background fields extracted
  from the whole resume at upload — so the user confirms or corrects them instead
  of restating them.
- Skill Assessment gets **retrieved evidence** — the top-3 resume passages for a
  query built from the section's purpose and the confirmed target roles.

Both arrive in the system prompt as separate, explicitly unconfirmed blocks, after
KNOWN SO FAR and never inside it. Nothing here writes `user_data`: the existing
extraction of the user's own replies stays the only path into confirmed state.

Everything degrades. A failed lookup or retrieval is logged and the turn proceeds
exactly as it would without a resume. A conversation without a resume gets a
byte-identical system prompt, because no block is rendered at all.

Caching, because the router runs twice per turn:
  * the status lookup runs once per user message (`checked_for`);
  * retrieval runs once per (document, section, query) (`evidence_key`), and an
    empty result is never cached, so a failed retrieval is retried next turn;
  * a different document_id discards everything cached for the old one.
"""

import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from ..enums import SectionID
from ..models import ResumeContext, ResumeEvidence, XBuddyData

logger = logging.getLogger(__name__)

RESUME_SECTIONS = frozenset({SectionID.BACKGROUND, SectionID.SKILL_ASSESSMENT})
BACKGROUND_FIELDS = ("current_role", "years_experience", "highest_education", "work_history")
SKILL_EVIDENCE_K = 3

_FIELD_LABELS = {
    "current_role": "Current role",
    "years_experience": "Years of experience (stated in the resume)",
    "highest_education": "Education",
    "work_history": "Work history",
}

BACKGROUND_BLOCK = """FROM YOUR RESUME — NOT CONFIRMED
The user uploaded a resume. The values below were read from it. They are the
resume's claims, not the user's answers: they are not part of KNOWN SO FAR and
nothing here is recorded until the user confirms or corrects it.
{facts}

HOW TO USE THEM
- Do not ask the user to restate these. Present them together as one short
  summary in your own words and ask whether it is accurate or what to change.
  That is a single question.
- If the user corrects anything, the correction wins.
- Ask in the normal way for any Background field not listed here. Do not work
  out a missing value yourself — in particular, never calculate years of
  experience from dates; if it is not listed, ask for it as its own question."""

EVIDENCE_BLOCK = """RESUME EVIDENCE — NOT USER-CONFIRMED
Passages retrieved from the user's resume as possible evidence for this section.
They show what the resume says, not what the user claims about themselves.
{passages}

HOW TO USE THEM
- You may point to concrete work, projects, tools, or results these passages
  actually contain, and ask whether the user counts them as strengths or current
  skills.
- Use only what the passages say. Do not infer skills from job titles and do not
  add anything the passages do not contain. Ignore a passage that is not relevant.
- A skill is recorded only when the user says so. Their self-assessment decides."""

# Injected so tests steer them; production uses the store and the retrieval helper.
StatusLookup = Callable[[int, str], Awaitable[Any]]
Retrieve = Callable[[int, str, str, int], Awaitable[list]]


async def _fetch_status(user_id: int, thread_id: str):
    from .store import ResumeStore

    return await ResumeStore().status(user_id=user_id, thread_id=thread_id)


async def _retrieve(user_id: int, thread_id: str, query: str, k: int) -> list:
    from .retrieval import retrieve_resume_evidence

    return await retrieve_resume_evidence(user_id, thread_id, query, k=k)


def resume_rag_configured() -> bool:
    """Without Supabase credentials the feature is simply off — no lookups, no noise."""
    from core.settings import settings

    has_key = any(
        getattr(settings, name, None) is not None
        for name in ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    )
    return bool(getattr(settings, "SUPABASE_URL", None)) and has_key


def coerce_resume_context(value: Any) -> ResumeContext | None:
    """Checkpoints can hand back a dict; anything unusable is treated as absent."""
    if value is None or isinstance(value, ResumeContext):
        return value
    try:
        return ResumeContext.model_validate(value)
    except Exception:  # noqa: BLE001 - a corrupt cache entry must not break a turn
        logger.warning("resume context: discarding an unreadable cached value")
        return None


def latest_human_key(messages: list[BaseMessage]) -> str:
    """Identify the user message a lookup serves. Position is the fallback for id-less messages."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index].id or f"#{index}"
    return ""


def skill_query(user_data: XBuddyData) -> str:
    """The Skill Assessment retrieval query: the section's purpose, plus target roles once confirmed."""
    base = "Evidence of technical skills, tools, projects, and measurable achievements"
    roles = [role for role in user_data.target_roles if role and role.strip()]
    return f"{base} relevant to {', '.join(roles)}" if roles else base


def evidence_key(document_id: str, section: SectionID, query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}:{section.value}:{digest}"


@dataclass(frozen=True)
class ResumeResolution:
    """The context to render this turn, and whether state needs updating."""

    context: ResumeContext | None
    changed: bool


async def resolve_resume_context(
    *,
    user_id: int | None,
    thread_id: str | None,
    section: SectionID,
    user_data: XBuddyData,
    messages: list[BaseMessage],
    cached: Any,
    fetch_status: StatusLookup | None = None,
    retrieve: Retrieve | None = None,
) -> ResumeResolution:
    """Resolve this turn's resume context. Never raises."""
    current = coerce_resume_context(cached)

    # Only the two resume sections pay for a lookup; everywhere else the cache is
    # left exactly as it is and nothing is rendered.
    if section not in RESUME_SECTIONS or not user_id or not thread_id or not resume_rag_configured():
        return ResumeResolution(None, False)

    lookup = fetch_status or _fetch_status
    search = retrieve or _retrieve
    key = latest_human_key(messages)

    context = current
    if context is None or context.checked_for != key:
        try:
            status = await lookup(user_id, thread_id)
        except Exception:
            logger.exception("resume context: status lookup failed for thread %s; continuing without it",
                             str(thread_id)[:8])
            return ResumeResolution(None, False)

        if status is None:
            context = ResumeContext(document_id=None, checked_for=key)
        elif current is not None and current.document_id == status.document_id:
            context = current.model_copy(update={
                "checked_for": key,
                "filename": status.filename,
                "candidate_facts": status.candidate_facts,
            })
        else:
            # A new or replaced resume: nothing cached for the old one survives.
            context = ResumeContext(
                document_id=status.document_id,
                filename=status.filename,
                candidate_facts=status.candidate_facts,
                checked_for=key,
            )

    if context.document_id and section is SectionID.SKILL_ASSESSMENT:
        query = skill_query(user_data)
        wanted = evidence_key(context.document_id, section, query)
        if context.evidence_key != wanted:
            try:
                found = await search(user_id, thread_id, query, SKILL_EVIDENCE_K)
            except Exception:
                logger.exception("resume context: retrieval failed for thread %s; continuing without evidence",
                                 str(thread_id)[:8])
                found = []
            evidence = [
                ResumeEvidence(chunk_index=c.chunk_index, section=str(getattr(c.section, "value", c.section)),
                               content=c.content, similarity=c.similarity)
                for c in found
            ]
            # An empty result is almost always a failure (every resume has a
            # non-Summary chunk); caching it would disable evidence for the rest of
            # the section. Leave the key unset so the next turn tries again.
            context = context.model_copy(update={
                "evidence": evidence,
                "evidence_key": wanted if evidence else None,
            })

    return ResumeResolution(context, context != current)


def _is_empty(value: Any) -> bool:
    return value is None or value == [] or value == ""


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if str(item).strip())
    return str(value)


def render_resume_block(
    section: SectionID, context: ResumeContext | None, user_data: XBuddyData
) -> str | None:
    """The conditional prompt block for this section, or None for no block at all."""
    if context is None or not context.document_id:
        return None

    if section is SectionID.BACKGROUND and context.candidate_facts:
        # Only fields the user has not already confirmed: a confirmed value is never
        # re-proposed, and a correction is never overwritten by the resume.
        lines = [
            f"- {_FIELD_LABELS[name]}: {_render_value(context.candidate_facts.get(name))}"
            for name in BACKGROUND_FIELDS
            if _is_empty(getattr(user_data, name, None))
            and not _is_empty(context.candidate_facts.get(name))
        ]
        return BACKGROUND_BLOCK.format(facts="\n".join(lines)) if lines else None

    if section is SectionID.SKILL_ASSESSMENT and context.evidence:
        # Evidence is shown only for the exact document, section, and query it was
        # retrieved for — never carried into another section or a changed query.
        if context.evidence_key != evidence_key(context.document_id, section, skill_query(user_data)):
            return None
        passages = [e for e in context.evidence if e.section != "summary"]  # the RPC excludes it too
        if not passages:
            return None
        rendered = "\n".join(f"[{n}] {e.content.strip()}" for n, e in enumerate(passages, start=1))
        return EVIDENCE_BLOCK.format(passages=rendered)

    return None
