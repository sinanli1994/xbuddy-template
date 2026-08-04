"""Router node — handles section navigation and context loading.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/router.py

Fully deterministic: the navigation decision is already represented in
structured state (`router_directive` + `section_states`), so no LLM is involved.
Producing the directive from conversation is generate_decision's job (PR 3).

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


def _resolve_section(
    directive: Any,
    current_section: SectionID,
    section_states: dict[str, SectionState],
) -> _Resolution:
    """Apply the directive deterministically."""
    directive_str = directive.value if isinstance(directive, RouterDirective) else directive

    if not isinstance(directive_str, str):
        logger.warning("Unrecognized router_directive %r; normalizing to stay", directive)
        return _Resolution(current_section, normalize_to_stay=True)

    normalized = directive_str.strip().lower()

    if normalized == RouterDirective.STAY.value:
        return _Resolution(current_section)

    if normalized == RouterDirective.NEXT.value:
        # Next *unfinished*, not next in sequence: on a cold start the directive
        # is NEXT while current_section is already CAREER_GOAL, so advancing by
        # sequence would skip the first section entirely.
        target = get_next_unfinished_section(section_states)
        if target is None:
            logger.info("All sections complete; marking conversation finished")
            return _Resolution(current_section, all_done=True)
        return _Resolution(target)

    if normalized.startswith(_MODIFY_PREFIX):
        target = _parse_modify_target(normalized)
        if target is None:
            logger.warning(
                "Invalid modify target in %r; normalizing directive to stay", directive_str
            )
            return _Resolution(current_section, normalize_to_stay=True)
        return _Resolution(target, is_valid_modify=True)

    logger.warning("Unrecognized router_directive %r; normalizing to stay", directive_str)
    return _Resolution(current_section, normalize_to_stay=True)


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

    was_finished = bool(state.get("finished", False))
    if resolution.all_done:
        update["finished"] = True
    elif was_finished and resolution.is_valid_modify:
        # A valid modify on a finished workflow makes it live again. Section
        # statuses and content stay as they are; memory_updater (PR 4) owns
        # moving the reopened section off DONE when its content changes.
        logger.info("Reopening finished conversation at section %s", section.value)
        update["finished"] = False

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

    update["context_packet"] = build_context_packet(
        section_id=section,
        status=active.status,
        draft=active.content,
        user_data=state.get("user_data") or XBuddyData(),
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
