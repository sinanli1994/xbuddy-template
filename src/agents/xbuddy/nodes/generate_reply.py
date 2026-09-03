"""Generate reply node — creates the user-facing conversational response.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_reply.py

Streaming note: this node calls `ainvoke`, not `astream`, and never touches a
token. `ChatOpenAI(streaming=True)` fires token callbacks during `ainvoke`, and
LangGraph surfaces them under `stream_mode="messages"`, which the service turns
into SSE `token` events. Assembling chunks here would duplicate that stream for
no benefit.

Ordinary replies are deliberately untagged and stream live. Structured first-draft
proposals are private until validation, then emit one complete message. Decision,
extraction, and raw proposal JSON must never appear in the user's token stream.
"""

import logging
import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..action_plan import propose_first_draft
from ..enums import SectionID
from ..models import ContextPacket, PendingActionPlan, XBuddyData, XBuddyState
from ..sections.base_prompt import SATISFACTION_OVERLAY

logger = logging.getLogger(__name__)

# How many recent messages to send with each reply. Keeps context bounded as a
# conversation grows; the checkpoint still holds the full history.
SHORT_MEMORY_SIZE = 10

# Used when the router did not run, so no system prompt exists. Deliberately not
# an LLM call: prompting a model with no section context produces worse output
# than admitting the problem.
FALLBACK_REPLY = (
    "Sorry — I lost track of where we were in your job search plan. "
    "Could you tell me again what you're looking for next?"
)


def _reply_model():
    """Resolve the model for replies.

    Indirection exists so tests can patch one function instead of reaching into
    core.llm. `get_model()` is @cache'd and shared across nodes, so per-call
    behaviour must go through config rather than mutating the instance.
    """
    from core.llm import get_model

    return get_model()


def _build_messages(
    state: XBuddyState, packet: ContextPacket
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Return (messages to send, the trimmed window) for the current turn.

    The packet is passed in rather than re-read from state so the caller's
    None-check is the single place that guard lives.

    The system prompt comes from the context packet the router built in PR 2 —
    this node never re-assembles prompts. When a satisfaction handshake is open,
    a turn-scoped overlay is appended so the agent acknowledges the user's
    confirmation instead of asking another collection question.
    """
    system_prompt = packet.system_prompt

    awaiting_satisfaction = bool(state.get("awaiting_satisfaction_feedback", False))
    if awaiting_satisfaction:
        system_prompt = f"{system_prompt}\n\n{SATISFACTION_OVERLAY.strip()}"

    history: list[BaseMessage] = list(state.get("messages", []))
    window: list[BaseMessage] = history[-SHORT_MEMORY_SIZE:]
    return [SystemMessage(content=system_prompt), *window], window


async def generate_reply_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    """Generate the conversational reply for the current section."""
    logger.info(
        "[REPLY] start monotonic_ms=%.0f thread=%s section=%s intent=%s",
        time.monotonic() * 1000, state.get("thread_id"),
        state.get("current_section"), state.get("reply_intent", "CONVERSE"),
    )
    packet = state.get("context_packet")
    if packet is None:
        logger.error("generate_reply: no context_packet; returning fallback without calling model")
        fallback: dict[str, Any] = {
            "messages": [AIMessage(content=FALLBACK_REPLY)],
            "awaiting_user_input": True,
            "last_error": "context_packet missing in generate_reply",
        }
        return fallback  # type: ignore[return-value]

    messages, window = _build_messages(state, packet)

    if state.get("reply_intent") == "PROPOSE_FIRST_DRAFT":
        data = state.get("user_data") or XBuddyData()
        try:
            if packet.section_id is not SectionID.ACTION_PLAN or data.action_items:
                raise ValueError("proposal intent requires Action Plan with no agreed items")
            content, steps = await propose_first_draft(packet, data, config)
            message_id = str(uuid4())
            reply = AIMessage(content=content, id=message_id)
            proposal_update: dict[str, Any] = {
                "messages": [reply], "short_memory": window,
                "pending_action_plan": PendingActionPlan(message_id=message_id, action_items=steps),
                "awaiting_user_input": True,
                "awaiting_satisfaction_feedback": True,
            }
        except Exception as exc:
            logger.exception("generate_reply: structured proposal failed validation")
            proposal_update = {
                "messages": [AIMessage(content=(
                    "I couldn't prepare a valid action-plan draft this time. "
                    "Please retry and I'll use the information you've already shared."
                ))],
                "awaiting_user_input": True,
                "last_error": f"action plan proposal error: {exc}",
                "error_count": state.get("error_count", 0) + 1,
            }
        return proposal_update  # type: ignore[return-value]

    try:
        response = await _reply_model().ainvoke(messages, config)
    except Exception as exc:  # degrade to a reply rather than kill the turn
        logger.exception("generate_reply: model call failed")
        degraded: dict[str, Any] = {
            "messages": [AIMessage(content=FALLBACK_REPLY)],
            "awaiting_user_input": True,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"generate_reply model error: {exc}",
        }
        return degraded  # type: ignore[return-value]

    content = response.content if isinstance(response.content, str) else str(response.content)
    logger.info(
        "generate_reply: section=%s chars=%d overlay=%s",
        state.get("current_section"),
        len(content),
        state.get("awaiting_satisfaction_feedback", False),
    )

    # Only the new message is returned — `messages` carries the add_messages
    # reducer, so returning the history would duplicate it. No additional_kwargs:
    # section_id/agent_name there break langchain_to_chat_message.
    update: dict[str, Any] = {
        "messages": [AIMessage(content=content)],
        "short_memory": window,
        "awaiting_user_input": True,
    }
    return update  # type: ignore[return-value]
