"""Pydantic models for the JobBuddy Agent.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/models.py
"""

from typing import Any, NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator

from .enums import SectionID, SectionStatus


class SectionContent(BaseModel):
    """Content for an agent section."""
    content: dict[str, Any]  # Rich text content (Tiptap JSON format)
    plain_text: str | None = None  # Plain text version for LLM processing


class SectionState(BaseModel):
    """State of a single section."""
    section_id: SectionID
    content: SectionContent | None = None
    satisfaction_status: str | None = None  # satisfied, needs_improvement, or None
    status: SectionStatus = SectionStatus.PENDING


class ContextPacket(BaseModel):
    """Context packet loaded by the router for the current section."""
    section_id: SectionID
    status: SectionStatus
    system_prompt: str
    draft: SectionContent | None = None
    validation_rules: dict[str, Any] | None = None


class XBuddyData(BaseModel):
    """Job-search data collected across the five JobBuddy sections.

    Intentionally flat and minimal: the LLM extraction that populates these
    fields arrives in a later PR, so the schema stays easy to widen later.

    Every field is defaulted, so `XBuddyData()` is valid and partial
    mid-conversation data never fails validation. Combined with Pydantic's
    default `extra="ignore"`, this keeps the model compatible with older
    checkpoints in both directions.
    """

    # --- Career Goal ---
    target_roles: list[str] = Field(default_factory=list)
    career_goal_summary: str | None = None
    target_timeline: str | None = None  # free text: "3 months", "ASAP"

    # --- Background ---
    current_role: str | None = None
    years_experience: int | None = None
    highest_education: str | None = None
    work_history: list[str] = Field(default_factory=list)

    # --- Job Preferences ---
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_work_modes: list[str] = Field(default_factory=list)  # "remote", "hybrid", "onsite"
    target_industries: list[str] = Field(default_factory=list)
    employment_types: list[str] = Field(default_factory=list)  # "full-time", "contract"
    salary_expectation: str | None = None  # free text, avoids currency modeling

    # --- Skill Assessment ---
    strengths: list[str] = Field(default_factory=list)
    current_skills: list[str] = Field(default_factory=list)
    skill_gaps: list[str] = Field(default_factory=list)

    # --- Action Plan ---
    action_items: list[str] = Field(default_factory=list)


class ChatAgentDecision(BaseModel):
    """Structured decision from the generate_decision node."""
    router_directive: str = Field(
        ...,
        description="Navigation control: 'stay', 'next', or 'modify:<section_id>'",
    )
    user_satisfaction_feedback: str | None = Field(
        None, description="User's feedback about satisfaction with the section."
    )
    is_satisfied: bool | None = Field(
        None, description="Whether the user is satisfied with the current section."
    )
    should_save_content: bool = Field(
        False,
        description="Whether to save the current section content.",
    )

    @field_validator("router_directive")
    def validate_router_directive(cls, v):
        if v not in ["stay", "next"] and not v.startswith("modify:"):
            raise ValueError("router_directive must be 'stay', 'next', or 'modify:<section_id>'")
        return v


class ChatAgentOutput(BaseModel):
    """Complete output from the generate_reply + generate_decision nodes."""
    reply: str = Field(..., description="Conversational response to the user.")
    router_directive: str = Field(
        ...,
        description="Navigation control: 'stay', 'next', or 'modify:<section_id>'",
    )
    user_satisfaction_feedback: str | None = None
    is_satisfied: bool | None = None
    should_save_content: bool = False

    @field_validator("router_directive")
    def validate_router_directive(cls, v):
        if v not in ["stay", "next"] and not v.startswith("modify:"):
            raise ValueError("router_directive must be 'stay', 'next', or 'modify:<section_id>'")
        return v


class XBuddyState(MessagesState):
    """State for the JobBuddy agent.

    Extends MessagesState (which provides `messages: list[BaseMessage]`).

    IMPORTANT: MessagesState is a TypedDict, not a Pydantic BaseModel, so
    class-level defaults (including `Field(default_factory=...)`) are inert —
    they are never applied at runtime. `initialize_node` is therefore the
    single source of state defaults; see nodes/initialize.py and
    state_factory.py.

    Every key added here is `NotRequired`, because between START and the end
    of `initialize` the state legitimately contains almost nothing, and nodes
    return partial update dicts rather than whole states.

    `messages` carries the `add_messages` reducer, so it must never be
    returned by a node that did not intend to append.
    """

    # User and conversation identification
    user_id: NotRequired[int]
    thread_id: NotRequired[str]

    # Navigation and progress
    current_section: NotRequired[SectionID]
    context_packet: NotRequired[ContextPacket | None]
    section_states: NotRequired[dict[str, SectionState]]
    router_directive: NotRequired[str]
    finished: NotRequired[bool]

    # Domain-specific data
    user_data: NotRequired[XBuddyData]

    # Memory management
    short_memory: NotRequired[list[BaseMessage]]

    # Agent output
    agent_output: NotRequired[ChatAgentOutput | None]
    awaiting_user_input: NotRequired[bool]
    awaiting_satisfaction_feedback: NotRequired[bool]

    # Error tracking
    error_count: NotRequired[int]
    last_error: NotRequired[str | None]

    # Final output — the generated job search strategy
    final_output: NotRequired[str | None]
    should_generate_final_output: NotRequired[bool]
