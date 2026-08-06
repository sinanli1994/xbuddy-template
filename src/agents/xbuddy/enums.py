"""Enumerations for your XBuddy Agent."""

from enum import Enum, StrEnum


class SectionStatus(str, Enum):
    """Status of an agent section."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class RouterDirective(str, Enum):
    """Router directive for navigation control."""
    STAY = "stay"
    NEXT = "next"
    MODIFY = "modify"  # Format: "modify:section_id"


class DecisionAction(StrEnum):
    """What the decision model chose to do about the active section.

    Kept separate from RouterDirective: the model picks one of these three plus
    a target section, and generate_decision composes the `modify:<id>` string.
    The model never has to produce that syntax itself.

    Uses StrEnum rather than the `(str, Enum)` pattern of the other enums in this
    module. Those predate this PR and are left alone; new code should prefer
    StrEnum, whose str() returns the value instead of "DecisionAction.STAY".
    """
    STAY = "stay"
    NEXT = "next"
    MODIFY = "modify"


class SectionID(str, Enum):
    """JobBuddy's section identifiers.

    Declaration order is load-bearing: `get_next_section` and
    `get_next_unfinished_section` in prompts.py walk `list(SectionID)`,
    so these must stay in the order the user progresses through them.
    """
    CAREER_GOAL = "career_goal"
    BACKGROUND = "background"
    JOB_PREFERENCES = "job_preferences"
    SKILL_ASSESSMENT = "skill_assessment"
    ACTION_PLAN = "action_plan"
