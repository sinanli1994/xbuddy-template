"""Process a summary response before choosing the one visible reply.

Ordinary collection turns still stream then extract as before. Confirmation turns
must decide and commit navigation first: asking a reply model in the old section
to simulate the next section is not a transition.
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ..models import XBuddyState
from .generate_decision import generate_decision_node


def latest_input_id(state: XBuddyState) -> str | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.id
    return None


def confirmation_processed(state: XBuddyState) -> bool:
    input_id = latest_input_id(state)
    return input_id is not None and state.get("confirmation_processed_id") == input_id


async def process_confirmation_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    update: dict[str, Any] = dict(await generate_decision_node(state, config))
    update["confirmation_processed_id"] = latest_input_id(state)
    return update  # type: ignore[return-value]
