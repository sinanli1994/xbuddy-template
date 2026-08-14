"""Memory updater node — captures what the turn revealed about the user.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/memory_updater.py

PR 4 stages 2-5:

* **Stage 2** — section-scoped structured extraction into `XBuddyData`. The first
  thing in the codebase that *writes* `user_data`, which is what makes the
  `KNOWN SO FAR` block in PR 2's prompts stop being empty.
* **Stage 3** — the `DONE` transition and `should_generate_final_output`. Before
  this, nothing marked a section complete, so `next` could never advance and PR
  3's reply cap was the only thing bounding the loop.
* **Stages 4-5** — the durable Supabase record, with a retry queue for writes
  that fail.

Four properties matter most:

* **Its tokens must never reach the user.** The extraction call is tagged
  `internal_extraction`, which the service already drops at service.py:762 —
  the same mechanism verified for `internal_decision` in PR 3.
* **It never raises and never forgets.** Every failure path leaves `user_data`
  exactly as it was. Extraction can add and correct; it cannot silently drop
  what the user already said.
* **Satisfaction, extraction, and persistence are independent.** Section
  progress is computed before either model or network call, so neither a failed
  extraction nor a failed write can withhold a completion the user confirmed.
* **One error per turn.** Reasons are accumulated and reported together, so
  `error_count` advances at most once no matter how many things failed. See
  `memory_updater_node` for the precedence rule.

State may legitimately run ahead of the durable record: the checkpoint is
primary for a live thread, and `persistence_pending` is what keeps that
divergence observable rather than silent.
"""

import logging
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from ..enums import SectionID, SectionStatus
from ..extraction import extraction_changed, get_extract_model, merge_extraction
from ..models import ContextPacket, SectionState, XBuddyData, XBuddyState
from ..persistence import persist_section
from ..sections.base_prompt import EXTRACTION_RULES
from ..state_factory import coerce_section_state

logger = logging.getLogger(__name__)

# Tag the service uses to keep this call's tokens out of the user's stream.
INTERNAL_EXTRACTION_TAG = "internal_extraction"

EXTRACTION_WINDOW_SIZE = 10


def _extraction_chain(extract_model: type[BaseModel]):
    """Build the structured-output chain for one section's schema.

    Indirection exists so tests patch one function. `.with_config(tags=...)`
    propagates to the inner LLM run, which is what suppresses these tokens at
    the SSE layer.

    `method="json_schema"` requires an OpenAI-family model. `get_model()` is
    typed as a union that also includes ChatGroq (function_calling/json_mode
    only) and ChatVertexAI (json_mode only), so mypy flags the argument. The
    default is OpenAIModelName.GPT_4O, and PR 3's decision node carries exactly
    the same requirement — it escapes the error only because it passes a
    concrete schema class rather than a `type[BaseModel]`, which resolves a
    different overload. Switching LLMConfig.DEFAULT_MODEL to Groq or Vertex
    would break both nodes at runtime; that is a pre-existing constraint of the
    structured-output design, not something this node introduces.
    """
    from core.llm import get_model

    return (
        get_model()
        .with_structured_output(
            extract_model,
            method="json_schema",  # type: ignore[arg-type]
            strict=True,
            include_raw=True,
        )
        .with_config(tags=[INTERNAL_EXTRACTION_TAG])
    )


def _error_update(state: XBuddyState, reason: str) -> dict[str, Any]:
    """The guard exit, used only when no work at all can be attempted.

    Everything downstream of the guard accumulates reasons instead and reports
    them once at the end of the turn.
    """
    logger.warning("memory_updater: %s", reason)
    return {
        "last_error": reason,
        "error_count": state.get("error_count", 0) + 1,
    }


def _section_progress(state: XBuddyState, section_id: SectionID) -> dict[str, Any]:
    """Apply the DONE transition and the completion flag, as a partial update.

    The gate is `agent_output.is_satisfied is True` and nothing else. PR 3
    defined that operationally: a summary was presented *and* the user affirmed
    it. `False` means they asked for changes, `None` means no handshake has
    happened — neither completes a section.

    Only the **active** section can transition, and `DONE` is never downgraded,
    so a `modify:` revisit keeps its record and a later `next` still skips it.

    Computed independently of extraction and persistence: satisfaction is a
    separate signal, so neither a model failure nor a database outage may
    silently withhold a completion the user asked for.
    """
    output = state.get("agent_output")
    existing = state.get("section_states") or {}
    sections = {key: coerce_section_state(value) for key, value in existing.items()}

    update: dict[str, Any] = {}

    if output is not None and output.is_satisfied is True:
        active = sections.get(section_id.value)
        if active is None:
            active = SectionState(section_id=section_id, status=SectionStatus.PENDING)
            sections[section_id.value] = active
        if active.status is not SectionStatus.DONE:
            sections[section_id.value] = active.model_copy(
                update={"status": SectionStatus.DONE}
            )
            logger.info("memory_updater: section %s marked done", section_id.value)

    # New mapping, never a mutation of the checkpointed one; emitted only when it
    # actually differs, matching the router's rule.
    if sections != existing:
        update["section_states"] = sections

    # Every one of the five, present and done. The per-section check matters:
    # `all()` over a partial mapping would be vacuously true.
    all_done = all(
        (entry := sections.get(section.value)) is not None
        and entry.status is SectionStatus.DONE
        for section in SectionID
    )
    if all_done:
        logger.info("memory_updater: all sections done; final output can be generated")
        update["should_generate_final_output"] = True

    return update


def _with_progress(progress: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Combine section progress with another partial update.

    A plain function rather than an inline `{**a, **b}` at each return: mypy
    rejects `**` expansion where a TypedDict is expected, and this keeps the
    merge in one place so no exit path can forget the progress half.
    """
    merged: dict[str, Any] = {**progress, **extra}
    return merged


def _known_values(packet: ContextPacket, user_data: XBuddyData) -> str:
    """Render what is already stored for this section's fields.

    Given to the model so it can distinguish "unchanged" from "corrected". An
    echoed identical value is harmless — the merge is idempotent.
    """
    required = (packet.validation_rules or {}).get("required_fields", [])
    lines = []
    for field_name in required:
        value = getattr(user_data, field_name, None)
        rendered = "not yet provided" if value is None or value == [] else repr(value)
        lines.append(f"- {field_name}: {rendered}")
    return "\n".join(lines) if lines else "- (this section declares no required fields)"


def _build_prompt(
    packet: ContextPacket, user_data: XBuddyData, messages: list[BaseMessage]
) -> list[BaseMessage]:
    """The packet is passed in so the caller's None-check is the only guard."""
    context = (
        f"CURRENT SECTION: {packet.section_id.value}\n\n"
        f"ALREADY STORED FOR THIS SECTION\n{_known_values(packet, user_data)}"
    )
    window = messages[-EXTRACTION_WINDOW_SIZE:]
    return [SystemMessage(content=f"{EXTRACTION_RULES.strip()}\n\n{context}"), *window]


async def _extract(
    packet: ContextPacket,
    before: XBuddyData,
    messages: list[BaseMessage],
    config: RunnableConfig,
) -> tuple[XBuddyData | None, str | None]:
    """Run extraction. Returns (merged_data_or_None, error_reason_or_None).

    `None` data means "nothing to apply" — either the model found nothing new or
    it failed. The reason distinguishes those: a no-op carries no reason, a
    failure does. Never raises.
    """
    try:
        extract_model = get_extract_model(packet.section_id)
    except ValueError as exc:
        return None, f"no extraction schema: {exc}"

    try:
        result = await _extraction_chain(extract_model).ainvoke(
            _build_prompt(packet, before, messages), config
        )
    except Exception as exc:  # a failed extraction must not kill the turn
        logger.exception("memory_updater: extraction model call failed")
        return None, f"extraction model error: {exc}"

    parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
    extracted = result.get("parsed") if isinstance(result, dict) else None

    if parsing_error is not None or not isinstance(extracted, extract_model):
        return None, f"unparseable extraction output: {parsing_error}"

    after = merge_extraction(extracted, before)
    if not extraction_changed(before, after):
        logger.info("memory_updater: section=%s no new data", packet.section_id.value)
        return None, None

    changed_fields = [
        name
        for name in type(extracted).model_fields
        if getattr(before, name, None) != getattr(after, name, None)
    ]
    logger.info(
        "memory_updater: section=%s updated=%s", packet.section_id.value, changed_fields
    )
    return after, None


def _dedupe(values: list[str]) -> list[str]:
    """Order-preserving unique.

    The queue holds each section at most once, which also bounds it: there are
    only five sections, so it can never exceed five entries however many turns
    fail in a row.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


async def _persist(
    state: XBuddyState,
    packet: ContextPacket,
    sections: dict[str, SectionState],
    user_data: XBuddyData,
    current_is_dirty: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Retry queued writes, then persist the current section if warranted.

    Returns (partial_update, error_reasons); never raises.

    Queued sections go **first** so one that has been failing for several turns
    is not starved behind the active section. The current section is skipped in
    the retry loop and handled once at the end, so it is never written twice in
    a single run.
    """
    user_id = int(state.get("user_id", 1))
    thread_id = str(state.get("thread_id", ""))
    current_value = packet.section_id.value

    # `raw` is what state actually holds; `queued` is what we act on. They differ
    # when state carries duplicates, and the comparison at the end uses `raw` so
    # a de-duplicated queue is written back rather than left dirty forever.
    raw = list(state.get("persistence_pending") or [])
    queued = _dedupe(raw)
    still_pending: list[str] = []
    errors: list[str] = []

    async def attempt(section_value: str) -> bool:
        section = sections.get(section_value)
        if section is None:
            # A queued section absent from state has nothing to write. Drop it
            # rather than retrying forever.
            logger.warning(
                "memory_updater: queued section %s absent from state; dropping",
                section_value,
            )
            return True
        return await persist_section(user_id, thread_id, section_value, section, user_data)

    for section_value in queued:
        if section_value == current_value:
            continue
        if await attempt(section_value):
            logger.info("memory_updater: retry persisted %s", section_value)
        else:
            still_pending.append(section_value)
            errors.append(f"persistence retry failed for section {section_value}")

    # The current section, when its durable row would differ or it is already
    # queued from an earlier turn.
    if current_is_dirty or current_value in queued:
        if await attempt(current_value):
            logger.info("memory_updater: persisted %s", current_value)
        else:
            still_pending.append(current_value)
            errors.append(f"persistence failed for section {current_value}")

    update: dict[str, Any] = {}
    resolved = _dedupe(still_pending)
    if resolved != raw:
        update["persistence_pending"] = resolved

    return update, errors


async def memory_updater_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    """Capture the turn: extract facts, complete sections, persist the record.

    Error contract, deliberately chosen:

    * **One increment per turn.** `error_count` advances at most once however
      many extraction or persistence attempts failed — it counts bad turns, not
      bad rows, so a retry queue draining three stale sections cannot inflate it.
    * **Reasons are joined, not replaced.** `last_error` carries every reason
      from the turn, so a successful persistence never hides an extraction
      failure that happened alongside it, and vice versa. Neither wins; both are
      reported.
    * **A clean turn leaves `last_error` alone.** Nothing clears it, matching
      PR 1's initialize, which preserves it across turns. It is a "last known
      error", not "current status".
    """
    # Guard: without a packet we know neither which schema to extract with nor
    # which section to complete, so this exit skips progress and persistence too.
    packet = state.get("context_packet")
    if packet is None:
        logger.error("memory_updater: no context_packet; not calling the model")
        return _error_update(state, "context_packet missing in memory_updater")  # type: ignore[return-value]

    progress = _section_progress(state, packet.section_id)
    errors: list[str] = []

    before = state.get("user_data") or XBuddyData()
    messages: list[BaseMessage] = list(state.get("messages", []))

    # Nothing said yet means nothing to extract. Not a failure — a turn with no
    # new facts, which still gets its progress and persistence applied.
    merged: XBuddyData | None = None
    if messages:
        merged, extraction_error = await _extract(packet, before, messages, config)
        if extraction_error is not None:
            errors.append(extraction_error)
    else:
        logger.debug("memory_updater: empty message history; nothing to extract")

    effective_data = merged if merged is not None else before
    sections = progress.get("section_states") or {
        key: coerce_section_state(value)
        for key, value in (state.get("section_states") or {}).items()
    }

    # The current section's durable row differs when its status changed or when
    # extraction produced new content for it.
    #
    # `agent_output.should_save_content` is deliberately NOT the gate here, even
    # though it is available on the same state. DECISION_RULES never mentions it
    # (see sections/base_prompt.py), so the model emits it with no rule to apply —
    # it is advisory, not authoritative. Durable writes follow actual state
    # changes instead: this is a fact about whether the row would differ, not a
    # judgement about whether it deserves saving. The two legitimately disagree
    # in both directions, and the tests named
    # `test_persists_despite_should_save_content_false` and
    # `test_does_not_persist_despite_should_save_content_true` pin that as
    # intended rather than accidental.
    current_is_dirty = "section_states" in progress or merged is not None

    try:
        persistence_update, persistence_errors = await _persist(
            state, packet, sections, effective_data, current_is_dirty
        )
    except Exception as exc:  # bookkeeping must not kill the turn either
        logger.exception("memory_updater: persistence bookkeeping failed")
        persistence_update, persistence_errors = {}, [f"persistence error: {exc}"]
    errors.extend(persistence_errors)

    update = _with_progress(progress, persistence_update)
    if merged is not None:
        update["user_data"] = merged

    if errors:
        update["last_error"] = "; ".join(errors)
        update["error_count"] = state.get("error_count", 0) + 1

    return update  # type: ignore[return-value]
