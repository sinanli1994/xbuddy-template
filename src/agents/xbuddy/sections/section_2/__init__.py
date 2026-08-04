"""Section 2 — Background.

Captures where the user is coming from, so the gap between their history and
their Section 1 target can be assessed later in Skill Assessment.
"""

from ...enums import SectionID
from ..base_prompt import SectionTemplate, ValidationRule

BACKGROUND_PROMPT = """
CURRENT SECTION: Background

You have the user's target role. Now establish where they are starting from.
You need four things:

1. current_role — what they do now, or did most recently. If they are between
   jobs, that is fine; record the last role and move on without dwelling on it.
2. years_experience — total relevant years, as a number. An approximation is
   fine; "about eight" becomes 8.
3. highest_education — highest completed qualification, or the most relevant
   one. "No degree" is a complete answer and needs no follow-up.
4. work_history — the roles that matter for the target, one line each:
   company or context, role, rough dates. Two or three entries is usually
   plenty; you do not need their entire CV.

HOW TO GET THERE
- Start with what they do now, then work backwards only as far as is relevant
  to the target role from Section 1.
- Ask for years of experience as its own question. Do not infer it from dates.
- If a gap or a career break comes up, take it at face value, record it in one
  neutral line, and keep going. Do not probe for a reason.
- If they list a long history, ask which two or three roles they think matter
  most for where they are heading, rather than transcribing all of it.
- Achievements will be covered under Skill Assessment. If they volunteer one
  here, note it briefly and say you will come back to it.

WHEN YOU HAVE ALL FOUR
Summarize the current role, years of experience, education, and the roles you
recorded, then ask whether that is accurate before moving to preferences.
"""

SECTION_2_TEMPLATE = SectionTemplate(
    section_id=SectionID.BACKGROUND,
    name="Background",
    description=(
        "Capture the user's current role, years of experience, education, and "
        "the work history relevant to their target."
    ),
    system_prompt_template=BACKGROUND_PROMPT,
    validation_rules=[
        ValidationRule(
            field_name="current_role",
            rule_type="required",
            value=True,
            error_message="A current or most recent role is needed to assess the gap.",
        ),
        ValidationRule(
            field_name="years_experience",
            rule_type="required",
            value=True,
            error_message="Years of experience is needed to calibrate role seniority.",
        ),
        ValidationRule(
            field_name="work_history",
            rule_type="min_length",
            value=1,
            error_message="At least one work history entry is needed.",
        ),
    ],
    required_fields=[
        "current_role",
        "years_experience",
        "highest_education",
        "work_history",
    ],
    next_section=SectionID.JOB_PREFERENCES,
)
