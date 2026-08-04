"""Ten Career Goal test cases, shared by both prompt variants.

Both variants see identical inputs so the comparison is apples-to-apples. Each
case carries the `known` fields already collected, which is what lets the
`progress_signal` evaluator know which field the reply should be driving at.
"""

from typing import Any

DATASET_NAME = "jobbuddy-section1-career-goal"

# `known` maps to XBuddyData fields already collected; `unfilled` is what the
# section still needs. `notes` explains what the case is probing.
CASES: list[dict[str, Any]] = [
    {
        "id": "vague_opener",
        "history": [],
        "user_message": "I need a new job.",
        "known": {},
        "unfilled": ["target_roles", "career_goal_summary", "target_timeline"],
        "notes": "Cold open with nothing to work from. Baseline case.",
    },
    {
        "id": "dont_know",
        "history": [("ai", "What kind of role are you looking for next?")],
        "user_message": "Honestly I have no idea. I just know I can't stay here.",
        "known": {},
        "unfilled": ["target_roles", "career_goal_summary", "target_timeline"],
        "notes": "Must not push for a title. Should pivot to likes/dislikes.",
    },
    {
        "id": "field_not_role",
        "history": [("ai", "What kind of role are you looking for next?")],
        "user_message": "I want to get into AI.",
        "known": {},
        "unfilled": ["target_roles", "career_goal_summary", "target_timeline"],
        "notes": "A field, not a role. Should ask what they'd do day to day.",
    },
    {
        "id": "career_changer",
        "history": [("ai", "What kind of role are you looking for next?")],
        "user_message": "I've been a nurse for nine years and I want out of healthcare entirely.",
        "known": {},
        "unfilled": ["target_roles", "career_goal_summary", "target_timeline"],
        "notes": "No target yet, strong push signal. Must not assume a destination.",
    },
    {
        "id": "over_specific",
        "history": [],
        "user_message": (
            "I'm targeting a Staff Platform Engineer role at a mid-size fintech, "
            "ideally signed within four months."
        ),
        "known": {},
        "unfilled": ["career_goal_summary"],
        "notes": "Role and timeline already given. Should ask only about the why.",
    },
    {
        "id": "competing_roles",
        "history": [("ai", "What kind of role are you looking for next?")],
        "user_message": "Either engineering manager or staff engineer. I genuinely can't decide.",
        "known": {},
        "unfilled": ["career_goal_summary", "target_timeline"],
        "notes": "Two valid targets. Recording both is fine; should not force a pick.",
    },
    {
        "id": "no_timeline",
        "history": [("ai", "What's driving the move?")],
        "user_message": "I'm bored and I've stopped learning anything.",
        "known": {"target_roles": ["Senior Data Engineer"]},
        "unfilled": ["career_goal_summary", "target_timeline"],
        "notes": "Role known. Should capture the why and move to timing.",
    },
    {
        "id": "unrealistic_timeline",
        "history": [("ai", "When would you like to be in the new role?")],
        "user_message": "I want to be a CTO by next month. I'm a junior dev right now.",
        "known": {"target_roles": ["CTO"]},
        "unfilled": ["career_goal_summary", "target_timeline"],
        "notes": "Should say why it's unrealistic once, then ask what has give.",
    },
    {
        "id": "asks_back",
        "history": [("ai", "What kind of role are you looking for next?")],
        "user_message": "What do you think I should go for?",
        "known": {"current_role": "QA Analyst", "years_experience": 4},
        "unfilled": ["target_roles", "career_goal_summary", "target_timeline"],
        "notes": "Must not invent a target for them; should redirect with one question.",
    },
    {
        "id": "volunteers_later_section",
        "history": [("ai", "What kind of role are you looking for next?")],
        "user_message": (
            "Senior backend. Also I only want remote, minimum 95k, and I'm strongest "
            "in Go and Postgres."
        ),
        "known": {},
        "unfilled": ["career_goal_summary", "target_timeline"],
        "notes": "Section-discipline probe: preferences and skills belong to 3 and 4.",
    },
]


def as_langsmith_examples() -> tuple[list[dict], list[dict]]:
    """Split CASES into (inputs, outputs) for langsmith dataset creation."""
    inputs = [
        {
            "case_id": case["id"],
            "history": case["history"],
            "user_message": case["user_message"],
            "known": case["known"],
        }
        for case in CASES
    ]
    outputs = [
        {"unfilled": case["unfilled"], "notes": case["notes"]} for case in CASES
    ]
    return inputs, outputs
