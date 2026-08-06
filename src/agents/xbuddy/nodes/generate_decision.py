"""Generate decision node — analyzes the conversation and sets the directive.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_decision.py

Machine-facing counterpart to generate_reply. It produces no user-visible text:
its structured output only sets `router_directive` for the router to act on.

Two properties matter most here:

* **Its tokens must never reach the user.** `json_schema` structured output comes
  back as ordinary message content, so without the `internal_decision` tag the
  raw JSON would stream straight into the SSE feed. The service drops tagged
  chunks at service.py:748.
* **It never navigates.** It writes `router_directive` and nothing else about
  position; `current_section`, `section_states`, and `finished` belong to the
  router (PR 2) and memory_updater (PR 4).

Structured output uses `method="json_schema", strict=True, include_raw=True`,
verified against the live API before this node was written.
"""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..enums import DecisionAction, RouterDirective, SectionID
from ..models import (
    ChatAgentDecision,
    ChatAgentOutput,
    ContextPacket,
    SectionDecision,
    XBuddyData,
    XBuddyState,
)
from ..sections.base_prompt import DECISION_RULES

logger = logging.getLogger(__name__)

# Tag the service uses to keep this call's tokens out of the user's stream.
INTERNAL_DECISION_TAG = "internal_decision"

# Replies the agent may produce for a single user message before the decision
# node stops asking the model and forces `stay`. Two allows the intended
# "finish a section, greet the next" turn without letting the
# reply -> decision -> memory_updater -> router loop run away.
MAX_REPLIES_PER_TURN = 2

DECISION_WINDOW_SIZE = 10


def _decision_chain():
    """Build the structured-output chain.

    Indirection exists so tests patch one function. `.with_config(tags=...)`
    propagates to the inner LLM run, which is what suppresses these tokens at
    the SSE layer — verified, not assumed.
    """
    from core.llm import get_model

    return (
        get_model()
        .with_structured_output(
            SectionDecision, method="json_schema", strict=True, include_raw=True
        )
        .with_config(tags=[INTERNAL_DECISION_TAG])
    )


def _replies_since_last_human(messages: Sequence[BaseMessage]) -> int:
    """Count AI replies produced since the user last spoke."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            count += 1
    return count


def _last_reply_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def _missing_required_fields(state: XBuddyState) -> list[str]:
    """Which of the section's required fields are still empty.

    Computed deterministically rather than left to the model to infer — the
    model is told what is missing, and only judges the conversation.
    """
    packet = state.get("context_packet")
    if packet is None or not packet.validation_rules:
        return []
    required = packet.validation_rules.get("required_fields", [])
    user_data = state.get("user_data") or XBuddyData()
    missing = []
    for field_name in required:
        value = getattr(user_data, field_name, None)
        if value is None or value == [] or value == "":
            missing.append(field_name)
    return missing


def _resolve_awaiting(
    state: XBuddyState, decision: SectionDecision | None, *, guard_path: bool
) -> bool:
    """Decide the value of awaiting_satisfaction_feedback.

    Written on every run so the flag can never go stale: a confirmed section that
    stayed True would re-trigger the reply overlay and ask the user to confirm
    something they already confirmed.

    On guard paths there is no evidence about the handshake, so an already-open
    one is preserved rather than silently closed.
    """
    if guard_path or decision is None:
        return bool(state.get("awaiting_satisfaction_feedback", False))

    if decision.action in (DecisionAction.NEXT, DecisionAction.MODIFY):
        return False
    return bool(decision.presented_summary) and not bool(decision.is_satisfied)


def _build_update(
    state: XBuddyState,
    directive: str,
    decision: SectionDecision | None,
    *,
    guard_path: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    """Assemble the node's update. Every exit path goes through here.

    Guards, fallbacks, and the happy path all produce the same shape, so a safe
    exit can never accidentally emit navigation fields or messages.
    """
    output = ChatAgentOutput(
        reply=_last_reply_text(list(state.get("messages", []))),
        router_directive=directive,
        user_satisfaction_feedback=decision.user_satisfaction_feedback if decision else None,
        is_satisfied=decision.is_satisfied if decision else None,
        should_save_content=bool(decision.should_save_content) if decision else False,
    )

    update: dict[str, Any] = {
        "router_directive": directive,
        "agent_output": output,
        "awaiting_satisfaction_feedback": _resolve_awaiting(state, decision, guard_path=guard_path),
    }
    if error is not None:
        update["last_error"] = error
    return update


def _stay_update(state: XBuddyState, *, reason: str, error: bool = False) -> dict[str, Any]:
    """Safe exit: hold the section and let route_decision end the turn.

    One flag, deliberately. `error=True` means a real system failure, and it
    both records `last_error` and increments `error_count` — the two can no
    longer drift. They previously sat behind separate flags, which let three of
    the four failure paths record an error without ever counting it.

    `error=False` is for outcomes that are normal control flow rather than
    failures: the reply cap is the only one, and it neither counts nor records.

    Every safe exit routes through here, so error accounting lives in exactly
    one place instead of being repeated per branch.
    """
    logger.info("generate_decision: forcing stay (%s)", reason)
    update = _build_update(
        state,
        RouterDirective.STAY.value,
        None,
        guard_path=True,
        error=reason if error else None,
    )
    if error:
        update["error_count"] = state.get("error_count", 0) + 1
    return update


def _compose_directive(decision: SectionDecision) -> str:
    """Map the structured action onto a router_directive string.

    The model picks an enum and a target; this composes `modify:<id>` so the
    model never has to produce that syntax. An unusable modify degrades to stay
    rather than guessing a destination.
    """
    if decision.action is DecisionAction.STAY:
        return RouterDirective.STAY.value
    if decision.action is DecisionAction.NEXT:
        return RouterDirective.NEXT.value

    target = decision.modify_target
    if target is None:
        logger.warning("generate_decision: modify without a target; falling back to stay")
        return RouterDirective.STAY.value
    try:
        return f"modify:{SectionID(target).value}"
    except ValueError:
        logger.warning("generate_decision: invalid modify target %r; falling back to stay", target)
        return RouterDirective.STAY.value


def _build_prompt(state: XBuddyState, packet: ContextPacket) -> list[BaseMessage]:
    """The packet is passed in so the caller's None-check is the only guard."""
    missing = _missing_required_fields(state)
    missing_text = ", ".join(missing) if missing else "none — every required field has a value"

    context = (
        f"CURRENT SECTION: {packet.section_id.value}\n"
        f"SECTION STATUS: {packet.status.value}\n"
        f"REQUIRED FIELDS STILL EMPTY: {missing_text}\n"
        f"VALID SECTIONS FOR modify_target: "
        f"{', '.join(section.value for section in SectionID)}"
    )

    history: list[BaseMessage] = list(state.get("messages", []))
    window = history[-DECISION_WINDOW_SIZE:]
    return [SystemMessage(content=f"{DECISION_RULES.strip()}\n\n{context}"), *window]


async def generate_decision_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    """Analyze the conversation and produce a structured navigation decision."""
    messages: list[BaseMessage] = list(state.get("messages", []))

    # Guard 1: reply cap. Checked before the model so a capped iteration costs
    # neither a request nor a trace span.
    if _replies_since_last_human(messages) >= MAX_REPLIES_PER_TURN:
        return _stay_update(state, reason="reply cap reached for this turn")  # type: ignore[return-value]

    # Guard 2: no context packet means we do not know which section is active.
    packet = state.get("context_packet")
    if packet is None:
        logger.error("generate_decision: no context_packet; not calling the model")
        return _stay_update(state, reason="context_packet missing", error=True)  # type: ignore[return-value]

    try:
        result = await _decision_chain().ainvoke(_build_prompt(state, packet), config)
    except Exception as exc:  # a failed decision must not kill the turn
        logger.exception("generate_decision: model call failed")
        return _stay_update(  # type: ignore[return-value]
            state, reason=f"decision model error: {exc}", error=True
        )

    parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
    decision = result.get("parsed") if isinstance(result, dict) else None

    if parsing_error is not None or not isinstance(decision, SectionDecision):
        return _stay_update(  # type: ignore[return-value]
            state,
            reason=f"unparseable decision output: {parsing_error}",
            error=True,
        )

    directive = _compose_directive(decision)

    # Final gate against the same ChatAgentDecision validator the rest of the
    # codebase trusts.
    #
    # UNREACHABLE BY CONSTRUCTION, kept as defence in depth. _compose_directive
    # has five return statements: four yield the literals "stay"/"next", the
    # fifth an f-string prefixed "modify:". The validator accepts exactly those
    # three shapes, so the function is total with respect to it — verified
    # exhaustively over all 18 (action, modify_target) combinations a validated
    # SectionDecision can hold, and over hostile model_construct inputs that
    # bypass Pydantic entirely. Neither reaches the except branch.
    #
    # It stays because that proof rests on _compose_directive's current
    # implementation rather than on a type guarantee: an edit there could start
    # emitting something else, and this is what would catch it. Cost is one
    # discarded object per turn.
    #
    # Both claims above are pinned by tests, so this comment cannot quietly go
    # stale: test_composed_directive_is_always_valid covers the 18 validated
    # combinations, test_hostile_model_construct_inputs_stay_validator_safe
    # covers the bypassed ones.
    try:
        ChatAgentDecision(
            router_directive=directive,
            user_satisfaction_feedback=decision.user_satisfaction_feedback,
            is_satisfied=decision.is_satisfied,
            should_save_content=decision.should_save_content,
        )
    except ValueError:
        logger.warning(
            "generate_decision: %r rejected by validator; falling back to stay", directive
        )
        return _stay_update(state, reason=f"invalid directive {directive!r}", error=True)  # type: ignore[return-value]

    logger.info(
        "generate_decision: action=%s directive=%s satisfied=%s reason=%s",
        decision.action.value,
        directive,
        decision.is_satisfied,
        decision.decision_reason,
    )

    # No `messages`, and nothing about position — the router owns navigation.
    return _build_update(state, directive, decision)  # type: ignore[return-value]
