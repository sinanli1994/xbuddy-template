"""Shared fixtures for the PR 3 node tests.

The two generation nodes each resolve their model through one module-level
helper (`_reply_model` / `_decision_chain`). Tests patch that helper with a
recording fake, so nothing here touches the network and every test can assert
both *what* the model was sent and *whether it was called at all* — the latter
matters for the two short-circuit guards.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.context import build_context_packet
from agents.xbuddy.enums import DecisionAction, SectionID, SectionStatus
from agents.xbuddy.models import SectionDecision, XBuddyData
from agents.xbuddy.state_factory import build_initial_state


class RecordingModel:
    """Stands in for the chat model in generate_reply."""

    def __init__(self, reply: str = "What kind of role are you targeting next?"):
        self.reply = reply
        self.calls: list[list[Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0].content

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


class RecordingChain:
    """Stands in for the structured-output chain in generate_decision.

    Returns the include_raw shape: {"raw", "parsed", "parsing_error"}.
    """

    def __init__(
        self,
        decision: SectionDecision | None = None,
        parsing_error: Exception | None = None,
        raises: Exception | None = None,
    ):
        self.decision = decision
        self.parsing_error = parsing_error
        self.raises = raises
        self.calls: list[list[Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0].content

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        if self.raises is not None:
            raise self.raises
        return {
            "raw": AIMessage(content=""),
            "parsed": self.decision,
            "parsing_error": self.parsing_error,
        }


def _make_decision(**overrides) -> SectionDecision:
    """A valid SectionDecision with every required field supplied.

    The real schema has no defaults (strict json_schema requires all 7 fields in
    `required`), so tests must pass them all — this keeps that explicit.
    """
    values: dict[str, Any] = {
        "action": DecisionAction.STAY,
        "modify_target": None,
        "is_satisfied": None,
        "user_satisfaction_feedback": None,
        "should_save_content": False,
        "presented_summary": False,
        "decision_reason": "target_roles still empty",
    }
    values.update(overrides)
    return SectionDecision(**values)


def _make_state(
    *,
    section: SectionID = SectionID.CAREER_GOAL,
    messages: list | None = None,
    user_data: XBuddyData | None = None,
    with_packet: bool = True,
    **overrides,
) -> dict:
    """A realistic post-router state, as generate_reply/decision would receive it."""
    state = build_initial_state(user_id=7, thread_id="t-pr3")
    state["messages"] = (
        messages if messages is not None else [HumanMessage(content="I need a new job")]
    )
    state["current_section"] = section
    state["user_data"] = user_data or XBuddyData()
    state["context_packet"] = (
        build_context_packet(
            section_id=section,
            status=SectionStatus.IN_PROGRESS,
            user_data=state["user_data"],
        )
        if with_packet
        else None
    )
    state.update(overrides)
    return state


@pytest.fixture
def make_state():
    """Factory fixture — `tests/` is not a package, so helpers arrive this way."""
    return _make_state


@pytest.fixture
def make_decision():
    """Factory for a fully-populated SectionDecision (the schema has no defaults)."""
    return _make_decision


@pytest.fixture
def reply_model(monkeypatch):
    """Patch generate_reply's model and hand the fake back to the test."""
    from agents.xbuddy.nodes import generate_reply as module

    fake = RecordingModel()
    monkeypatch.setattr(module, "_reply_model", lambda: fake)
    return fake


@pytest.fixture
def decision_chain(monkeypatch):
    """Patch generate_decision's chain; tests set `.decision` / `.raises`."""
    from agents.xbuddy.nodes import generate_decision as module

    fake = RecordingChain(decision=_make_decision())
    monkeypatch.setattr(module, "_decision_chain", lambda: fake)
    return fake
