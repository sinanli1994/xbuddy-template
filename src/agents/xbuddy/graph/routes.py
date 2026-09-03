"""Routing logic for XBuddy Agent graph.

These functions are used as conditional edges in the StateGraph.
Study FounderBuddy's routes.py:
https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/graph/routes.py
"""

from typing import Literal

from langchain_core.messages import HumanMessage

from ..enums import RouterDirective
from ..models import XBuddyState
from ..nodes.process_confirmation import confirmation_processed


def route_turn(state: XBuddyState) -> Literal["process_confirmation", "generate_reply"] | None:
    target = route_decision(state)
    packet = state.get("context_packet")
    required = (packet.validation_rules or {}).get("required_fields", []) if packet else []
    data = state.get("user_data")
    ready_for_review = bool(required) and all(
        getattr(data, field, None) not in (None, [], "") for field in required
    )
    # Complete-field checkpoints written by an older build may have no handshake
    # flag. Let the decision model judge the previous summary + latest user input
    # before speaking; never depend on a yes/no phrase match or a stale flag.
    if target and not confirmation_processed(state) and (
        state.get("awaiting_satisfaction_feedback") or ready_for_review
    ):
        return "process_confirmation"
    return target


def route_after_reply(state: XBuddyState) -> Literal["generate_decision"] | None:
    # No second extraction/decision: it would apply the old confirmation to the
    # new section and could incorrectly mark the just-proposed plan as agreed.
    if confirmation_processed(state) or state.get("reply_intent") == "PROPOSE_FIRST_DRAFT":
        return None
    return "generate_decision"


def route_after_implementation(state: XBuddyState) -> Literal["router"] | None:
    # An existing artifact may only need a persistence retry. If finalization
    # produced no visible message, the still-pending user input needs its reply.
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], HumanMessage):
        return "router"
    return None


def route_after_memory_updater(state: XBuddyState) -> Literal["implementation", "router"]:
    """Route after memory_updater — generate final output or loop back.

    TODO: This checks should_generate_final_output. Make sure your
    memory_updater sets this flag when all sections are complete.
    """
    if state.get("should_generate_final_output", False):
        return "implementation"
    return "router"


def route_decision(state: XBuddyState) -> Literal["generate_reply"] | None:
    """Determine whether to generate a reply or end the turn.

    The router runs both before and after the reply/decision/memory cycle.  A
    navigation directive chooses which section is active; it is not permission
    to generate another user-facing message in the same invocation.  Once the
    newest message is an AI reply, the turn is complete even if ``next`` or
    ``modify`` moved the structured state to another section.

    An empty history is the one exception: it supports callers that invoke a
    fresh graph solely to obtain the opening greeting.
    """
    def has_pending_user_input() -> bool:
        msgs = state.get("messages", [])
        if not msgs:
            return False
        return isinstance(msgs[-1], HumanMessage)

    if state.get("finished", False):
        if has_pending_user_input():
            return "generate_reply"
        return None

    directive = state.get("router_directive")

    if directive == RouterDirective.STAY or (isinstance(directive, str) and directive.lower() == "stay"):
        if has_pending_user_input():
            return "generate_reply"
        if state.get("awaiting_user_input", False):
            return None
        return None

    elif directive == RouterDirective.NEXT or (
        isinstance(directive, str) and directive.startswith("modify:")
    ):
        if has_pending_user_input() or not state.get("messages", []):
            return "generate_reply"
        return None

    return None
