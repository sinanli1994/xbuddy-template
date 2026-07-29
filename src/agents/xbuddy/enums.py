"""Enumerations for your XBuddy Agent."""

from enum import Enum


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
