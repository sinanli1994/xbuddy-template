"""Implementation node — produces the final artifact once every section is done.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_business_plan.py

For JobBuddy the artifact is a job search strategy: one grounded synthesis call
(`synthesis.synthesize_final_output`), rendered to Markdown by
`final_output.render_final_output`, stored in `state["final_output"]`.

Three properties matter most:

* **Once only.** An existing `final_output` is the authoritative guard. It has to
  be, because `should_generate_final_output` is set whenever all five sections are
  DONE and is never cleared, so this node is re-entered on every later turn
  (`route_after_memory_updater`). Guarding on the flag would either regenerate the
  document every turn or require clearing a flag other nodes read.
* **The artifact does not go through the chat.** The synthesis call is tagged
  `internal_synthesis`, which the service drops at service.py:762, and this node
  appends one short readiness line rather than the document. The user reads it in
  the editor.
* **Never raises, never half-writes.** Any failure leaves `final_output` unset, so
  the next turn retries cleanly. A partially-written artifact would be
  indistinguishable from a complete one to every reader.

PR 5 Stage 3 covers the wiring only. Persisting the artifact to Supabase, and
invalidating it when a completed section is reopened, are later stages — so today a
`modify:` after completion leaves a stale document in place.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..enums import SectionID, SectionStatus
from ..final_output import render_final_output
from ..models import XBuddyState
from ..persistence import persist_final_output
from ..state_factory import coerce_section_state
from ..synthesis import synthesize_final_output

logger = logging.getLogger(__name__)

# The whole of what the chat says about the artifact. Deliberately one line: the
# document is long, structured, and lives in the editor, and pasting Markdown into
# a chat bubble is worse than pointing at it.
FINAL_OUTPUT_READY_MESSAGE = (
    "Your job search strategy is ready — you can open and edit it now."
)

# Values for the `final_output_pending` state key. One scalar rather than a queue:
# a thread has exactly one artifact, so there is never more than one outstanding
# durable operation for it. "write" is owned by this node; "stale" is owned by
# memory_updater's invalidation path.
FINAL_OUTPUT_PENDING_WRITE = "write"
FINAL_OUTPUT_PENDING_STALE = "stale"


def _error_update(state: XBuddyState, reasons: list[str]) -> dict[str, Any]:
    """One increment per failing turn, reasons joined.

    Mirrors `memory_updater`'s contract exactly: `error_count` counts bad turns
    rather than bad operations, and every reason from this node is reported together
    so one failure cannot mask another.

    `last_error` is replaced rather than appended to the previous value. Joining
    with whatever is already in state would resurrect a stale error from an earlier
    turn, because nothing clears `last_error` on success.
    """
    joined = "; ".join(reasons)
    logger.warning("implementation: %s", joined)
    return {
        "last_error": joined,
        "error_count": state.get("error_count", 0) + 1,
    }


async def _write_durable(state: XBuddyState, markdown: str) -> tuple[bool, str | None]:
    """Persist the artifact, translating identity out of state.

    Split out so the first-generation path and the retry path cannot drift.
    """
    user_id = int(state.get("user_id", 1))
    thread_id = str(state.get("thread_id", ""))
    return await persist_final_output(user_id, thread_id, markdown)


def _ineligibility_reasons(state: XBuddyState) -> list[str]:
    """Why this state cannot produce a final artifact. Empty list means eligible.

    Checked here rather than trusted from the router because this node is the only
    thing that writes the artifact, and a document synthesized from an incomplete
    conversation is worse than no document.

    The empty-plan case is an upstream invariant violation, not a shape the artifact
    should represent: Section 5 requires at least three agreed steps, and the plan is
    the document's spine. Synthesizing around an absent spine would produce something
    that looks finished and is not.
    """
    reasons: list[str] = []

    existing = state.get("section_states") or {}
    sections = {key: coerce_section_state(value) for key, value in existing.items()}
    not_done = [
        section.value
        for section in SectionID
        if (entry := sections.get(section.value)) is None
        or entry.status is not SectionStatus.DONE
    ]
    if not_done:
        reasons.append(f"final output requires every section done; still open: {not_done}")

    user_data = state.get("user_data")
    if user_data is None:
        reasons.append("final output requires user_data; none in state")
    elif not user_data.action_items:
        reasons.append(
            "final output requires confirmed action items; the Action Plan section "
            "completed with none"
        )

    return reasons


async def implementation_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    """Synthesize the final artifact, exactly once.

    State transition, in order:

    1. `final_output` already set -> `{}`. No model call, no message, no error.
    2. Ineligible -> `last_error` + `error_count += 1`. No artifact, no message.
    3. Synthesis fails -> same. `final_output` stays unset so a later turn retries.
    4. Success -> `{"final_output": <markdown>, "messages": [one AIMessage]}`.

    Never raises: the graph edge is `implementation -> END`, so an exception here
    would take down a turn whose conversational work has already succeeded.
    """
    existing_artifact = state.get("final_output")
    if existing_artifact:
        # The guard is about *synthesis*, not about persistence. An artifact whose
        # durable write was refused or failed still needs writing, and retrying it
        # costs nothing but one idempotent upsert — no model call, no new message.
        if state.get("final_output_pending") == FINAL_OUTPUT_PENDING_WRITE:
            persisted, reason = await _write_durable(state, existing_artifact)
            if persisted:
                logger.info("implementation: durable final output written on retry")
                cleared: dict[str, Any] = {"final_output_pending": None}
                return cleared  # type: ignore[return-value]
            return _error_update(state, [reason or "final output write failed"])  # type: ignore[return-value]

        logger.debug("implementation: final_output already exists; nothing to do")
        empty: dict[str, Any] = {}
        return empty  # type: ignore[return-value]

    reasons = _ineligibility_reasons(state)
    if reasons:
        return _error_update(state, reasons)  # type: ignore[return-value]

    user_data = state.get("user_data")
    assert user_data is not None  # guaranteed by _ineligibility_reasons

    try:
        final_output, error = await synthesize_final_output(user_data)
    except Exception as exc:  # synthesize_final_output should not raise; belt and braces
        logger.exception("implementation: synthesis raised unexpectedly")
        return _error_update(state, [f"synthesis raised: {exc}"])  # type: ignore[return-value]

    if final_output is None:
        return _error_update(state, [error or "synthesis produced no final output"])  # type: ignore[return-value]

    markdown = render_final_output(final_output)
    logger.info("implementation: final output generated (%d chars)", len(markdown))

    update: dict[str, Any] = {
        "final_output": markdown,
        "messages": [AIMessage(content=FINAL_OUTPUT_READY_MESSAGE)],
    }

    # The graph keeps the artifact regardless of what the database does. A refusal
    # to overwrite user edits is the case that matters most: the durable row stays
    # as the user left it, the conversation still gets its document, and the reason
    # is reported through the ordinary error contract rather than by discarding
    # work. Queue a retry only for genuine write failures — retrying a refusal
    # would just refuse again every turn.
    persisted, reason = await _write_durable(state, markdown)
    if not persisted:
        update["last_error"] = reason or "final output write failed"
        update["error_count"] = state.get("error_count", 0) + 1
        if reason and "has been edited" in reason:
            logger.warning("implementation: durable write refused to protect user edits")
        else:
            update["final_output_pending"] = FINAL_OUTPUT_PENDING_WRITE

    return update  # type: ignore[return-value]
