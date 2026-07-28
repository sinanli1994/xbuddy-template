"""Construction of JobBuddy state defaults.

`XBuddyState` extends `MessagesState`, which is a **TypedDict** — its
class-level `Field(default_factory=...)` declarations would be inert, so every
state default has to be produced explicitly. This module is that single source
of truth, shared by:

  * `nodes.initialize.initialize_node` — fills missing keys on every invocation
  * `initialize_xbuddy_state` — the cold-start factory used by the service layer

Keeping both paths on the same helpers means cold-start defaults cannot drift
apart from warm-start backfill.

It deliberately depends only on `.models` and `.enums` (never on `.nodes`), so
the service layer can import it without pulling in the graph.
"""

import uuid
from collections.abc import Callable
from typing import Any

from .enums import RouterDirective, SectionID, SectionStatus
from .models import SectionState, XBuddyData

# Default factory for every state key that is neither an identity field
# (user_id / thread_id) nor the section progress record (section_states).
#
# Factories, never shared literals: a mutable default must not be aliased
# across threads.
COLD_DEFAULTS: dict[str, Callable[[], Any]] = {
    # Navigation and progress
    "current_section": lambda: SectionID.CAREER_GOAL,
    "router_directive": lambda: RouterDirective.NEXT,
    "finished": lambda: False,
    # The router owns context_packet; initialize only ensures the key exists.
    "context_packet": lambda: None,
    # Domain data
    "user_data": XBuddyData,
    # Memory management
    "short_memory": list,
    # Agent output
    "agent_output": lambda: None,
    "awaiting_user_input": lambda: False,
    "awaiting_satisfaction_feedback": lambda: False,
    # Error tracking
    "error_count": lambda: 0,
    "last_error": lambda: None,
    # Final output
    "final_output": lambda: None,
    "should_generate_final_output": lambda: False,
}


def build_section_states() -> dict[str, SectionState]:
    """Return a fresh PENDING SectionState for every section, keyed by id value.

    Driven by iteration over `SectionID` rather than a hand-written list, so
    adding a section needs no change here. Keys are the enum *values*
    ("career_goal"), matching prompts.get_next_unfinished_section and the
    service layer's section_states lookups.
    """
    return {
        section_id.value: SectionState(section_id=section_id, status=SectionStatus.PENDING)
        for section_id in SectionID
    }


def coerce_section_state(value: Any) -> SectionState:
    """Coerce a checkpoint-restored section entry into a `SectionState`.

    Deserialized checkpoints can hand back plain dicts. Downstream consumers
    read attributes (e.g. `section_data.status.value` in the service layer), so
    a dict there would be an AttributeError.
    """
    if isinstance(value, SectionState):
        return value
    return SectionState.model_validate(value)


def merge_section_states(existing: Any) -> dict[str, SectionState]:
    """Preserve every existing section entry and backfill only missing ones.

    Never replaces or resets an entry: this is what lets a user leave halfway
    through and return without losing progress. `setdefault` (not assignment)
    guarantees existing content and satisfaction_status survive untouched.
    """
    merged: dict[str, SectionState] = {}
    if existing:
        merged = {key: coerce_section_state(value) for key, value in existing.items()}

    for section_id in SectionID:
        merged.setdefault(
            section_id.value,
            SectionState(section_id=section_id, status=SectionStatus.PENDING),
        )
    return merged


def build_initial_state(user_id: int = 1, thread_id: str | None = None) -> dict[str, Any]:
    """Build a complete cold-start state.

    `messages` is deliberately absent: it carries the `add_messages` reducer, so
    emitting it from a node would re-append and duplicate the conversation.
    """
    state: dict[str, Any] = {
        "user_id": user_id,
        "thread_id": thread_id or str(uuid.uuid4()),
        "section_states": build_section_states(),
    }
    for key, factory in COLD_DEFAULTS.items():
        state[key] = factory()
    return state


async def initialize_xbuddy_state(
    user_id: int | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Cold-start factory for a brand-new conversation.

    Called by the service layer when a request arrives with no thread_id, to
    mint a thread_id and report the starting section before the graph runs.
    Async to match that call site.
    """
    return build_initial_state(user_id=1 if user_id is None else user_id, thread_id=thread_id)
