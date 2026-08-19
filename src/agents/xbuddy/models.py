"""Pydantic models for the JobBuddy Agent.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/models.py
"""

from typing import Any, NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from .enums import DecisionAction, SectionID, SectionStatus


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


class CareerGoalExtract(BaseModel):
    """What the extraction model returns for the Career Goal section.

    Field names mirror that section's `required_fields` exactly, and every field
    is required-but-nullable — no Pydantic defaults. OpenAI's strict json_schema
    mode requires every key in `properties` to appear in `required`, and a field
    with a default is omitted from `required`, invalidating the schema. Verified
    against the live API for SectionDecision in PR 3; the same rule applies here.

    `None` means "no new information in this turn", never "clear the stored
    value" — see extraction.merge_extraction.
    """

    target_roles: list[str] | None = Field(
        description="Job titles the user is targeting. Null if not mentioned this turn."
    )
    career_goal_summary: str | None = Field(
        description="One or two sentences, in the user's words, on what this move should change."
    )
    target_timeline: str | None = Field(
        description="When they want to be in the new role, free text. Null if not mentioned."
    )


class BackgroundExtract(BaseModel):
    """Extraction schema for the Background section. See CareerGoalExtract."""

    current_role: str | None = Field(description="Their current or most recent role.")
    years_experience: int | None = Field(description="Total relevant years, as a whole number.")
    highest_education: str | None = Field(description="Highest or most relevant qualification.")
    work_history: list[str] | None = Field(
        description="Relevant roles, one line each: context, role, rough dates."
    )


class JobPreferencesExtract(BaseModel):
    """Extraction schema for the Job Preferences section. See CareerGoalExtract."""

    preferred_locations: list[str] | None = Field(description="Cities, regions, or 'remote'.")
    preferred_work_modes: list[str] | None = Field(description="remote / hybrid / on-site.")
    target_industries: list[str] | None = Field(description="Sectors they want to work in.")
    employment_types: list[str] | None = Field(
        description="full-time / part-time / contract / freelance / internship."
    )
    salary_expectation: str | None = Field(
        description="Free text in the user's own currency and period."
    )


class SkillAssessmentExtract(BaseModel):
    """Extraction schema for the Skill Assessment section. See CareerGoalExtract."""

    strengths: list[str] | None = Field(description="What they are good at, ideally with evidence.")
    current_skills: list[str] | None = Field(description="Concrete skills they can claim today.")
    skill_gaps: list[str] | None = Field(
        description="What the target role expects that they cannot yet evidence."
    )


class ActionPlanExtract(BaseModel):
    """Extraction schema for the Action Plan section. See CareerGoalExtract."""

    action_items: list[str] | None = Field(
        description="Concrete, individually startable steps the user agreed to."
    )


# Which extraction schema to use for each section. Mirrors SECTION_TEMPLATES:
# the field names of each model must equal that section's `required_fields`,
# which a test asserts so the two cannot drift.
EXTRACT_MODELS: dict[SectionID, type[BaseModel]] = {
    SectionID.CAREER_GOAL: CareerGoalExtract,
    SectionID.BACKGROUND: BackgroundExtract,
    SectionID.JOB_PREFERENCES: JobPreferencesExtract,
    SectionID.SKILL_ASSESSMENT: SkillAssessmentExtract,
    SectionID.ACTION_PLAN: ActionPlanExtract,
}


class SectionDecision(BaseModel):
    """Structured output returned by the decision model.

    EVERY field is required — no Pydantic defaults. Fields that may carry no
    value are required-but-nullable instead. OpenAI's strict json_schema mode
    requires every key in `properties` to appear in `required`, and a field with
    a default is omitted from `required`, which makes the schema invalid. With
    defaults, only 2 of these 7 fields would be required.

    Because nothing is optional, consumers must handle explicit nulls rather
    than relying on defaults.

    Verified against the live API with
    `with_structured_output(SectionDecision, method="json_schema", strict=True,
    include_raw=True)`: parsed round-trips and parsing_error is None.
    """

    action: DecisionAction = Field(
        description="Whether to stay in the current section, move to the next, or modify another."
    )
    modify_target: SectionID | None = Field(
        description="The section to jump to. Non-null only when action is 'modify'; null otherwise."
    )
    is_satisfied: bool | None = Field(
        description=(
            "True only if the agent presented a summary AND the user affirmed it. "
            "Null when no summary has been presented or the user has not responded to one."
        )
    )
    user_satisfaction_feedback: str | None = Field(
        description="What the user said about the summary, if anything. Null otherwise."
    )
    should_save_content: bool = Field(
        description="Whether the current section's content is worth persisting."
    )
    presented_summary: bool = Field(
        description="True if the agent's most recent reply presented a summary of the section."
    )
    decision_reason: str = Field(
        description=(
            "One sentence naming the concrete, observable signal that drove this choice "
            "(e.g. 'user confirmed the summary', 'target_timeline still empty'). "
            "Do not narrate deliberation."
        )
    )


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


class ActionItem(BaseModel):
    """One prioritized step in the final artifact.

    `step` is the user-facing instruction. When the Action Plan section produced
    it, this string is carried through **verbatim** — the user confirmed that
    wording, so synthesis may reorder or annotate it but never rewrite it. See
    final_output.py for the full contract.

    `rationale` is always synthesis, never the user's words. Keeping it in its own
    field is what lets the renderer label it as reasoning rather than as something
    the user said, which is the "recommendations stay distinguishable from user
    facts" half of the quality bar.

    No Pydantic defaults, deliberately: this model is LLM-facing, and OpenAI's
    strict json_schema mode requires every property to appear in `required`, which
    a default silently prevents. Same rule the extract models follow.
    """

    step: str = Field(
        description=(
            "The action, phrased so the user could start it today. If this came "
            "from a confirmed action item, reproduce that wording exactly."
        )
    )
    rationale: str = Field(
        description=(
            "Why this step, in one sentence, naming the collected fact it follows "
            "from. This is your reasoning, not the user's words."
        )
    )
    priority: int = Field(
        description="1-based rank. Across all items these must be exactly 1..n, each once."
    )
    timeframe: str | None = Field(
        description="When to do it, free text, or null when the user gave no timeline to size it against."
    )

    @field_validator("step", "rationale")
    @classmethod
    def reject_blank(cls, value: str, info: ValidationInfo) -> str:
        """A blank step or rationale is a structural defect, not a style opinion.

        An empty `step` renders as `**  **` and an empty `rationale` renders a bare
        "Why:" label — broken documents rather than merely weak ones. Deliberately
        narrow: nothing here judges whether the text is *good*, only that it exists.
        The value is not stripped, because the renderer's contract is that it never
        transforms content.
        """
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank or whitespace-only")
        return value


class FinalOutput(BaseModel):
    """The structured job search strategy, before rendering.

    Field order is the document's canonical section order — `render_final_output`
    follows it, so the two cannot drift.

    Every field is required and LLM-facing (see ActionItem on the no-defaults
    rule). List fields must be emitted as `[]` rather than omitted; an empty list
    means "nothing to say here", and for `unknowns` specifically it is an
    affirmative claim that nothing is missing, which the renderer states out loud.
    """

    headline: str = Field(
        description="One line naming the move: from what, to what, by when. No filler."
    )
    positioning_summary: str = Field(
        description=(
            "Two or three sentences on how this person should present themselves, "
            "built from their background, goal, and strengths."
        )
    )
    strengths_to_leverage: list[str] = Field(
        description="Strengths worth leading with, drawn from what the user reported."
    )
    skill_priorities: list[str] = Field(
        description="Gaps to close, most limiting first. Empty list if none were identified."
    )
    search_targets: list[str] = Field(
        description=(
            "Where to look — industries, company types, locations, or work modes — "
            "consistent with the stated preferences."
        )
    )
    action_items: list[ActionItem] = Field(
        description=(
            "The plan: exactly the confirmed Action Plan items, annotated. Assembled "
            "deterministically by synthesis.assemble_final_output — the model never "
            "authors this field, so a step cannot be dropped, reworded, or invented."
        )
    )
    risks_or_constraints: list[str] = Field(
        description=(
            "Real constraints the user stated — timeline pressure, location limits, "
            "time per week. Do not invent hazards."
        )
    )
    unknowns: list[str] = Field(
        description=(
            "What was never collected and would change this strategy. Say it here "
            "rather than guessing it anywhere else."
        )
    )

    @field_validator("action_items")
    @classmethod
    def priorities_form_a_total_order(cls, items: list[ActionItem]) -> list[ActionItem]:
        """Priorities must be exactly 1..n, each exactly once.

        Rejected rather than normalized on purpose. "Prioritized" is one of the
        seven quality dimensions, and 1,1,3 has no single correct reading — quietly
        renumbering it would invent an ordering the model did not express. Making
        this a model invariant also means the eval can assert grounding instead of
        spending a check on arithmetic.

        Stage 2 sees this as a `parsing_error` and degrades, exactly as PR 4's
        extraction does.
        """
        priorities = [item.priority for item in items]
        expected = list(range(1, len(items) + 1))
        if sorted(priorities) != expected:
            raise ValueError(
                f"action_item priorities must be exactly {expected}, got {sorted(priorities)}"
            )
        return items

    @field_validator("headline", "positioning_summary")
    @classmethod
    def reject_blank(cls, value: str, info: ValidationInfo) -> str:
        """See ActionItem.reject_blank. A blank headline renders as a bare `# `."""
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank or whitespace-only")
        return value


class ActionAnnotation(BaseModel):
    """What the model may add to one confirmed action item — and nothing more.

    `step_number` is the 1-based position of the item in the CONFIRMED ACTION PLAN
    block the model was shown. It is an alignment key, not a rank the model chooses:
    `synthesis.assemble_final_output` uses it to attach this annotation to the
    correct source string, then sets the real `priority` from the source order.

    The step text itself is deliberately absent. If the model were asked to echo it,
    "preserved exactly" would be a hope; here it is unrepresentable to get wrong.
    """

    step_number: int = Field(
        description="Which numbered item in CONFIRMED ACTION PLAN this annotates. 1-based."
    )
    rationale: str = Field(
        description=(
            "One sentence on why this step matters, naming the fact it follows from. "
            "Your reasoning, not the user's words."
        )
    )
    timeframe: str | None = Field(
        description=(
            "When to do it, sized to the user's stated timeline. Null if they gave no "
            "timeline — do not invent one."
        )
    )

    @field_validator("rationale")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        """Rejected here, at the model boundary, rather than during assembly.

        A blank rationale would otherwise pass draft validation and only fail when
        `assemble_final_output` builds the `ActionItem`, which reports it as an
        assembly problem. Catching it here attributes the failure to the model output
        where it belongs, and it surfaces as `parsing_error` so the caller degrades
        the same way as any other malformed response.
        """
        if not value.strip():
            raise ValueError("rationale must not be blank or whitespace-only")
        return value


class FinalOutputDraft(BaseModel):
    """What the synthesis model actually returns. Narrower than `FinalOutput`.

    Two fields of the final artifact are missing on purpose, because neither is the
    model's to decide:

    * **`action_items`** — assembled from the confirmed Action Plan plus these
      annotations, so step text and priority order are deterministic.
    * **`unknowns`** — derived from `XBuddyData` by `synthesis.derive_unknowns`. A
      model asked which facts are missing can both miss one and invent one; the
      state already knows the answer exactly.

    Every field is required with no defaults, for the strict json_schema reason
    documented on the extract models.
    """

    headline: str = Field(
        description="One line naming the move: from what, to what, by when. No filler."
    )
    positioning_summary: str = Field(
        description=(
            "Two or three sentences on how this person should present themselves, "
            "built only from the FACTS block."
        )
    )
    strengths_to_leverage: list[str] = Field(
        description="Strengths worth leading with, drawn from the FACTS block."
    )
    skill_priorities: list[str] = Field(
        description="Gaps to close, most limiting first. Empty list if none were collected."
    )
    search_targets: list[str] = Field(
        description=(
            "Where to look — industries, company types, locations, work modes — "
            "consistent with the stated preferences. Empty list if none were collected."
        )
    )
    action_annotations: list[ActionAnnotation] = Field(
        description=(
            "Exactly one annotation per numbered item in CONFIRMED ACTION PLAN, with "
            "step_number covering 1..n once each. Do not add or drop items."
        )
    )
    risks_or_constraints: list[str] = Field(
        description=(
            "Constraints the user actually stated — timeline pressure, location "
            "limits, time per week. Do not invent hazards."
        )
    )

    @field_validator("action_annotations")
    @classmethod
    def step_numbers_cover_each_item_once(
        cls, annotations: list[ActionAnnotation]
    ) -> list[ActionAnnotation]:
        """`step_number` must be exactly 1..n, each once.

        This is internal consistency only. Whether n matches the *actual* number of
        confirmed items is checked by `assemble_final_output`, which is the only place
        that knows the source data.
        """
        numbers = [annotation.step_number for annotation in annotations]
        expected = list(range(1, len(annotations) + 1))
        if sorted(numbers) != expected:
            raise ValueError(
                f"action_annotation step_numbers must be exactly {expected}, got {sorted(numbers)}"
            )
        return annotations


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

    # Sections whose Supabase write failed and should be retried on the next
    # memory_updater run. Canonical SectionID.value strings, each at most once.
    # State is allowed to run ahead of the durable record — the checkpoint is
    # primary for a live thread — and this queue is what keeps that divergence
    # observable instead of silent.
    persistence_pending: NotRequired[list[str]]

    # PR 5. Which durable final-output operation still needs doing, or None.
    # Values are the FINAL_OUTPUT_PENDING_* constants in nodes/implementation.py:
    # "write" (an artifact exists in state but its row was not written) or "stale"
    # (an artifact was retired but the row was not flagged).
    #
    # Deliberately NOT folded into `persistence_pending`, whose contract is
    # "canonical SectionID.value strings": memory_updater's retry loop looks each
    # entry up in `section_states` and silently drops anything it cannot find, so a
    # "final_output" token there would be treated as already-succeeded. One scalar
    # rather than a second queue — there is exactly one artifact per thread, so
    # there is nothing to queue.
    final_output_pending: NotRequired[str | None]

    # Final output — the generated job search strategy
    final_output: NotRequired[str | None]
    should_generate_final_output: NotRequired[bool]
