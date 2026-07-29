"""Initialize node — validates and sets up conversation state.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/initialize.py

This node runs once at the start of every invocation, so it must be
**idempotent**: it fills only what is missing and never overwrites existing
progress. That is what lets a user leave halfway through a conversation and
return later without losing anything.

  * Cold start (no checkpoint): populate every state key with its default.
  * Warm start (checkpoint restored): resolve identity from config, backfill
    missing section entries, and leave every other existing value untouched.
"""

import logging
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..enums import SectionStatus
from ..models import XBuddyState
from ..state_factory import COLD_DEFAULTS, merge_section_states

logger = logging.getLogger(__name__)


def _resolve_identity(state: XBuddyState, config: RunnableConfig | None) -> tuple[int, str]:
    """Resolve (user_id, thread_id) from config, then state, then defaults.

    Config wins: thread_id *is* the checkpoint key, and user_id comes from the
    authenticated request. `RunnableConfig` is itself a TypedDict and may be
    absent entirely (direct invocation, unit tests), so every access is a .get.
    """
    configurable = (config or {}).get("configurable") or {}

    thread_id = configurable.get("thread_id") or state.get("thread_id") or str(uuid.uuid4())

    # UserInput.user_id is `int | None`, and agent_config can inject values as
    # strings, so coerce rather than trust.
    raw_user_id = configurable.get("user_id", state.get("user_id"))
    try:
        user_id = 1 if raw_user_id is None else int(raw_user_id)
    except (TypeError, ValueError):
        logger.warning("Invalid user_id %r in config; defaulting to 1", raw_user_id)
        user_id = 1

    existing_user_id = state.get("user_id")
    if existing_user_id is not None and existing_user_id != user_id:
        logger.warning(
            "user_id mismatch for thread %s: config=%s checkpoint=%s; using config value",
            thread_id,
            user_id,
            existing_user_id,
        )

    return user_id, str(thread_id)


async def initialize_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    """Initialize conversation state, preserving any progress already made."""
    user_id, thread_id = _resolve_identity(state, config)

    update: dict[str, Any] = {"user_id": user_id, "thread_id": thread_id}

    # Absence, not falsiness: a legitimately False `finished` or 0 `error_count`
    # must not be mistaken for a missing key and overwritten.
    for key, factory in COLD_DEFAULTS.items():
        if key not in state:
            update[key] = factory()

    # Preserve every existing section entry; add only the ones that are missing.
    existing_sections = state.get("section_states")
    merged_sections = merge_section_states(existing_sections)
    if merged_sections != existing_sections:
        update["section_states"] = merged_sections

    is_cold_start = "section_states" not in state
    done_count = sum(
        1 for section in merged_sections.values() if section.status == SectionStatus.DONE
    )
    logger.info(
        "%s: user_id=%s thread_id=%s section=%s progress=%s/%s",
        "Cold start" if is_cold_start else "Warm start",
        user_id,
        thread_id,
        update.get("current_section", state.get("current_section")),
        done_count,
        len(merged_sections),
    )

    # `messages` is deliberately never returned: it carries the `add_messages`
    # reducer, so returning it would re-append and duplicate the history.
    return update  # type: ignore[return-value]
