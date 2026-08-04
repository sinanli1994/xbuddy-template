"""Section 1 — Career Goal.

Establishes what role the user wants next and on what timeline.

The prompt is deliberately split in two. `CAREER_GOAL_BODY` is the section's
subject matter; `CAREER_GOAL_QUESTIONING_STRATEGY` is the questioning approach
selected by the PR 2 A/B experiment. `evals/section1_career_goal/variants.py`
imports both constants — the body as its shared baseline and this strategy as its
`a_strict` arm — so the shipped prompt and the winning experiment arm cannot
drift apart.
"""

from ...enums import SectionID
from ..base_prompt import SectionTemplate, ValidationRule

CAREER_GOAL_BODY = """
CURRENT SECTION: Career Goal

Your job in this section is to turn a vague sense of "I need a new job" into a
target the rest of the plan can be built on. You need three things:

1. target_roles — the job titles they are aiming for. One is enough; two or
   three related ones are fine. "Something in tech" is not a role; keep going
   until you have a title someone could search for.
2. career_goal_summary — one or two sentences, in their words, on what they
   want this move to change. Money, title, autonomy, stability, and impact are
   all legitimate answers.
3. target_timeline — when they want to be in the new role. Free text is fine
   ("about three months", "before my visa expires in April").

HOW TO GET THERE
- Open by asking what they are looking for next, not why they are leaving.
- If the answer is a whole field rather than a role ("I want to get into AI"),
  ask what they picture themselves actually doing day to day, then name the
  closest one or two titles and ask if either fits.
- If they genuinely do not know, do not push for a title. Ask what they liked
  and disliked in their last role and work toward a title from there.
- If they name a role you would expect to require experience they have not
  mentioned, do not assume they lack it — ask.
- If the timeline is very tight for the move they are describing, say so once,
  explain briefly, and ask whether the date or the target has more give.

WHEN YOU HAVE ALL THREE
Summarize the role, the reason, and the timeline in three short lines, then ask
whether that is right before moving to their background.
"""

# Selected by the PR 2 A/B experiment as variant `a_strict`. The rejected arm
# (`b_anchored`, which offered example answers alongside the question) scored
# better on answerability but markedly worse on non-leading guidance (2.8 vs 4.6)
# and conciseness (3.1 vs 4.7), and broke the one-question rule in 40% of cases.
# Full results: evals/section1_career_goal/README.md.
CAREER_GOAL_QUESTIONING_STRATEGY = """
QUESTIONING STRATEGY
Ask exactly one question and stop. Do not offer examples, options, or
suggestions alongside it — the user's own words are what you want. If their
previous answer was vague, ask a narrower question rather than proposing
candidate answers.
"""

CAREER_GOAL_PROMPT = (
    f"{CAREER_GOAL_BODY.strip()}\n\n{CAREER_GOAL_QUESTIONING_STRATEGY.strip()}\n"
)

SECTION_1_TEMPLATE = SectionTemplate(
    section_id=SectionID.CAREER_GOAL,
    name="Career Goal",
    description=(
        "Establish the target role(s), why the user wants the move, and the "
        "timeline they are working to."
    ),
    system_prompt_template=CAREER_GOAL_PROMPT,
    validation_rules=[
        ValidationRule(
            field_name="target_roles",
            rule_type="required",
            value=True,
            error_message="At least one concrete job title is needed before moving on.",
        ),
        ValidationRule(
            field_name="career_goal_summary",
            rule_type="min_length",
            value=20,
            error_message="The goal summary should say what the user wants this move to change.",
        ),
        ValidationRule(
            field_name="target_timeline",
            rule_type="required",
            value=True,
            error_message="A target timeline is needed to size the action plan.",
        ),
    ],
    required_fields=["target_roles", "career_goal_summary", "target_timeline"],
    next_section=SectionID.BACKGROUND,
)
