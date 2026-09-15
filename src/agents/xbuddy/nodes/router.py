"""Router node — handles section navigation and context loading.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/router.py

Fully deterministic: the navigation decision is already represented in
structured state (`router_directive` + `section_states`), so no LLM is involved.
Producing the directive from conversation is generate_decision's job (PR 3).

The one piece of I/O here is Resume RAG context (resume/context.py): in
Background and Skill Assessment only, a resume status read and, for Skill
Assessment, one embedding plus a vector search. No model judges anything, the
result is cached in state so the second router pass of a turn repeats none of
it, and any failure degrades to "no resume context" rather than raising.

The router has two entry points — once per invocation from `initialize`, and
again on every loop back from `memory_updater` — so it must be safe to run
repeatedly within a single invocation.

Contract with graph/routes.py:route_decision, which reads state *after* this
node runs:
  * Valid directives are left untouched. Clearing a valid `next` to `stay` would
    end a cold-start turn before the agent ever greets the user.
  * Malformed directives are normalized to STAY. Leaving them in place is not
    equivalent: bare "modify", "garbage", and None all fall through
    route_decision to `return None` and silently end the turn.
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..context import build_context_packet
from ..enums import RouterDirective, SectionID, SectionStatus
from ..models import SectionState, XBuddyData, XBuddyState
from ..prompts import get_next_unfinished_section
from ..resume.context import render_resume_block, resolve_resume_context
from ..state_factory import coerce_section_state

logger = logging.getLogger(__name__)

_MODIFY_PREFIX = "modify:"


def _parse_modify_target(directive: str) -> SectionID | None:
    """Extract the target section from a `modify:<section_id>` directive.

    Returns None for a bare `modify`, an unknown section id, or a blank target —
    all of which the caller normalizes to STAY.
    """
    _, _, raw_target = directive.partition(":")
    raw_target = raw_target.strip()
    if not raw_target:
        return None
    try:
        return SectionID(raw_target)
    except ValueError:
        return None


@dataclass(frozen=True)
class _Resolution:
    """Outcome of applying a directive, so the caller never re-parses it."""

    section: SectionID
    all_done: bool = False
    normalize_to_stay: bool = False
    is_valid_modify: bool = False


def _stay(
    current_section: SectionID,
    section_states: dict[str, SectionState],
    *,
    normalize: bool = False,
) -> _Resolution:
    """`stay`, except on a section that is already finished.

    The production failure this exists for: the decision model confirmed Job
    Preferences with `is_satisfied` true but emitted `stay`. memory_updater marked
    the section DONE, the router kept it current, and the conversation sat in a
    completed section for the rest of its life — Skill Assessment never opened, so
    its resume retrieval never ran and the reply model answered "based on your
    resume" holding no resume at all.

    Completion is a fact about state rather than a matter of opinion, so the router
    treats it as one. `stay` still holds position while the section is unfinished,
    `next` and `modify` are untouched, and an all-done conversation still holds
    position for the final-output path.
    """
    active = section_states.get(current_section.value)
    if active is None or active.status is not SectionStatus.DONE:
        return _Resolution(current_section, normalize_to_stay=normalize)

    target = get_next_unfinished_section(section_states)
    if target is None:
        return _Resolution(current_section, all_done=True, normalize_to_stay=normalize)

    logger.info(
        "Section %s is done; advancing to %s despite a %s directive",
        current_section.value,
        target.value,
        "malformed" if normalize else "stay",
    )
    # `normalize` still applies: a malformed directive left in state would dead-end
    # route_decision, so it is rewritten to stay even though the section moved.
    return _Resolution(target, normalize_to_stay=normalize)


def _resolve_section(
    directive: Any,
    current_section: SectionID,
    section_states: dict[str, SectionState],
) -> _Resolution:
    """Apply the directive deterministically."""
    directive_str = directive.value if isinstance(directive, RouterDirective) else directive

    if not isinstance(directive_str, str):
        logger.warning("Unrecognized router_directive %r; normalizing to stay", directive)
        return _stay(current_section, section_states, normalize=True)

    normalized = directive_str.strip().lower()

    if normalized == RouterDirective.STAY.value:
        return _stay(current_section, section_states)

    if normalized == RouterDirective.NEXT.value:
        # Next *unfinished*, not next in sequence: on a cold start the directive
        # is NEXT while current_section is already CAREER_GOAL, so advancing by
        # sequence would skip the first section entirely.
        target = get_next_unfinished_section(section_states)
        if target is None:
            logger.info("All sections complete; holding on the current section")
            return _Resolution(current_section, all_done=True)
        return _Resolution(target)

    if normalized.startswith(_MODIFY_PREFIX):
        target = _parse_modify_target(normalized)
        if target is None:
            logger.warning(
                "Invalid modify target in %r; normalizing directive to stay", directive_str
            )
            return _stay(current_section, section_states, normalize=True)
        return _Resolution(target, is_valid_modify=True)

    logger.warning("Unrecognized router_directive %r; normalizing to stay", directive_str)
    return _stay(current_section, section_states, normalize=True)


async def router_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    """Route to the correct section and load its context."""
    current_section = SectionID(state.get("current_section", SectionID.CAREER_GOAL))
    directive = state.get("router_directive")

    existing_sections = state.get("section_states") or {}
    sections = {key: coerce_section_state(value) for key, value in existing_sections.items()}

    resolution = _resolve_section(directive, current_section, sections)
    section = resolution.section

    update: dict[str, Any] = {"current_section": section}

    if resolution.normalize_to_stay:
        # Required for the fallback to actually route like stay — see module docstring.
        update["router_directive"] = RouterDirective.STAY

    # The router no longer writes `finished` (Issue #10). It used to, but only from
    # inside the `next` branch, which made completion a property of whichever
    # directive the decision model happened to emit rather than of the sections
    # themselves. `memory_updater` derives it from `all_sections_complete`, which is
    # the same condition behind `should_generate_final_output`.
    #
    # Reopening is unaffected: a valid modify still routes to the target section, and
    # `finished` returns to False when memory_updater actually demotes it off DONE.
    # Until then the conversation is genuinely still complete, and `route_decision`
    # keeps replying because the user's modify message is pending input.
    was_finished = bool(state.get("finished", False))
    if was_finished and resolution.is_valid_modify:
        logger.info("Reopening finished conversation at section %s", section.value)

    # Mark the active section in progress. PENDING is the only status promoted:
    # DONE is never downgraded, so a revisit keeps its record and a later `next`
    # still skips it instead of looping.
    active = sections.get(section.value)
    if active is None:
        active = SectionState(section_id=section, status=SectionStatus.PENDING)
        sections[section.value] = active
    if active.status == SectionStatus.PENDING:
        active = active.model_copy(update={"status": SectionStatus.IN_PROGRESS})
        sections[section.value] = active
    if sections != existing_sections:
        update["section_states"] = sections

    data = state.get("user_data") or XBuddyData()

    # Resume RAG. Background and Skill Assessment may carry an unconfirmed resume
    # block; every other section, and every conversation without a resume, gets
    # none — so their prompt is unchanged. Never raises: a failed lookup or
    # retrieval is logged and the turn proceeds without resume context.
    resume = await resolve_resume_context(
        user_id=state.get("user_id"),
        thread_id=state.get("thread_id"),
        section=section,
        user_data=data,
        messages=list(state.get("messages", [])),
        cached=state.get("resume_context"),
    )
    if resume.changed:
        update["resume_context"] = resume.context

    update["context_packet"] = build_context_packet(
        section_id=section,
        status=active.status,
        draft=active.content,
        user_data=data,
        resume_block=render_resume_block(section, resume.context, data),
    )
    update["reply_intent"] = (
        "PROPOSE_FIRST_DRAFT"
        if section is SectionID.ACTION_PLAN and not data.action_items
        and not state.get("awaiting_satisfaction_feedback", False)
        else "CONVERSE"
    )

    logger.info(
        "Router: directive=%r section=%s status=%s finished=%s",
        directive,
        section.value,
        active.status.value,
        update.get("finished", was_finished),
    )

    # `messages` is never returned — it carries the add_messages reducer.
    return update  # type: ignore[return-value]
