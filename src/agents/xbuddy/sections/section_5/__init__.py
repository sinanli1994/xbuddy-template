"""Section 5 — Action Plan.

The final section. Turns everything collected so far into concrete, dated steps
the user can start on this week. `next_section` is None.
"""

from ...enums import SectionID
from ..base_prompt import SectionTemplate, ValidationRule

ACTION_PLAN_PROMPT = """
CURRENT SECTION: Action Plan

Last section. Everything above becomes a short list of steps the user can
actually start on. You need one thing, done well:

1. action_items — concrete, individually startable steps. Each one should name
   what to do, and where useful how often or by when.

WHAT MAKES AN ACTION ITEM USABLE
- Specific enough to start today: "Rewrite the CV summary to lead with the
  platform migration" — not "improve CV".
- Owned by the user, not by chance: "Message three former colleagues at target
  companies this week" — not "network more".
- Tied to something from the earlier sections: a gap from Skill Assessment, the
  timeline from Career Goal, a constraint from Job Preferences.

HOW TO GET THERE
- Propose a first draft yourself. You have their goal, background, preferences,
  and gaps — use them. Do not ask the user to invent the plan from nothing.
- Cover the obvious ground: application materials, the biggest skill gap, where
  they will find openings, and who they will talk to.
- Size the plan to the timeline they gave. A three-month search and a
  twelve-month one are not the same list.
- Then ask which items feel realistic and which do not, and adjust. One
  question at a time, as always.
- If they have very little time per week, cut the list rather than compressing
  it. Four real steps beat twelve aspirational ones.

WHEN THE PLAN IS AGREED
Summarize the final list in order of what to do first, and confirm they are
happy with it. This completes the last section, so the summary should read like
something they can act on immediately.
"""

SECTION_5_TEMPLATE = SectionTemplate(
    section_id=SectionID.ACTION_PLAN,
    name="Action Plan",
    description=(
        "Turn the goal, background, preferences, and skill gaps into concrete, "
        "prioritized next steps sized to the user's timeline."
    ),
    system_prompt_template=ACTION_PLAN_PROMPT,
    validation_rules=[
        ValidationRule(
            field_name="action_items",
            rule_type="min_length",
            value=3,
            error_message="An action plan needs at least three concrete steps.",
        ),
    ],
    required_fields=["action_items"],
    next_section=None,
)
