"""Structured first-draft proposals: the workflow chooses the mode, not the LLM.

Raw structured tokens are private. Only a validated proposal is rendered into
one checkpointed AI message. Proposed actions do NOT populate agreed action_items.
"""

import json

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field, field_validator

from .models import ContextPacket, XBuddyData


class ProposedAction(BaseModel):
    heading: str = Field(description="A short action heading, at most eight words; no numbering or Markdown.")
    action: str = Field(description="A concrete step the user can start; an instruction, not a question.")
    basis_fields: list[str] = Field(
        min_length=1,
        description="Names of nonempty FACTS fields that justify this step. Never invent a field.",
    )

    @field_validator("action")
    @classmethod
    def concrete_statement(cls, value: str) -> str:
        value = value.strip()
        if not 4 <= len(value.split()) <= 40 or len(value) > 300 or "?" in value or "\n" in value:
            raise ValueError("actions must be concise single-line statements, not questions")
        if "based on your" in value.casefold():
            raise ValueError("grounding belongs in basis_fields, not debug-like action prose")
        return value

    @field_validator("heading")
    @classmethod
    def short_heading(cls, value: str) -> str:
        value = value.strip()
        if not 1 <= len(value.split()) <= 8 or len(value) > 70 or any(c in value for c in "\n?*#[]"):
            raise ValueError("headings must be short plain-text labels")
        if "based on your" in value.casefold():
            raise ValueError("headings must not expose internal grounding")
        return value


class ActionPlanDraft(BaseModel):
    actions: list[ProposedAction] = Field(min_length=3)


def _proposal_chain():
    from core.llm import get_model

    return get_model().with_structured_output(
        ActionPlanDraft, method="json_schema", strict=True, include_raw=True,
    ).with_config(tags=["skip_stream"])


def proposal_facts(data: XBuddyData) -> dict:
    return {
        key: value for key, value in data.model_dump().items()
        if key != "action_items" and value is not None and value != [] and value != ""
    }


def render_proposal(draft: ActionPlanDraft, data: XBuddyData) -> str:
    # Revalidate even model_construct or a test double must obey the same gate.
    draft = ActionPlanDraft.model_validate(draft.model_dump())
    facts = proposal_facts(data)
    if len({item.action.casefold() for item in draft.actions}) != len(draft.actions):
        raise ValueError("proposal must contain distinct actions")
    lines = ["### First-Draft Action Plan", ""]
    for index, item in enumerate(draft.actions, 1):
        if any(field not in facts for field in item.basis_fields):
            raise ValueError("proposal cites an unknown or empty fact")
        # The provenance is validated above and stays internal; don't print facts
        # repeatedly as debug text. Keep step text unchanged for later confirmation.
        lines.extend([f"**{index}. {item.heading}**", "", f"- {item.action}", ""])
    lines.append("Which of these steps feel realistic, and what would you like to adjust?")
    return "\n".join(lines)


REVISED_PLAN_QUESTION = (
    "Does this revised action plan look right? "
    "If so, confirm it and I'll use it in your final career plan."
)


def render_revised_plan(steps: list[str]) -> str:
    """The user's revised plan, shown back verbatim for one explicit confirmation.

    Deterministic, like the proposal: a model asked to restate the plan could
    reword the very steps the user is about to confirm.
    """
    lines = ["### Revised Action Plan", ""]
    lines.extend(f"{index}. {step}" for index, step in enumerate(steps, 1))
    lines.extend(["", REVISED_PLAN_QUESTION])
    return "\n".join(lines)


async def propose_first_draft(packet: ContextPacket, data: XBuddyData, config) -> tuple[str, list[str]]:
    # Actual Section 5 template and all collected state, not a Skills transition overlay.
    message = SystemMessage(content=(
        packet.system_prompt + "\n\nREPLY MODE: PROPOSE_FIRST_DRAFT\n"
        "Return structured proposed actions covering application materials, a skill gap, "
        "finding openings, and people to contact. Each action must cite its FACTS field names. "
        "Give each step a short action heading and a concise action (at most 40 words). "
        "Prefer action + suggested timing/frequency + goal when the collected constraints support it. "
        "Timing is a proposal, not a claim about existing commitments. Do not invent commitments "
        "or qualifications, or repeat 'Based on your...' in headings or actions. "
        "Use only these user facts; recommendations are proposals, not invented user facts.\n"
        + "FACTS\n" + json.dumps(proposal_facts(data), ensure_ascii=False)
    ))
    result = await _proposal_chain().ainvoke([message], config)
    if result.get("parsing_error") or not isinstance(result.get("parsed"), ActionPlanDraft):
        raise ValueError("model did not return a valid ActionPlanDraft")
    draft = ActionPlanDraft.model_validate(result["parsed"].model_dump())
    content = render_proposal(draft, data)
    return content, [item.action for item in draft.actions]
