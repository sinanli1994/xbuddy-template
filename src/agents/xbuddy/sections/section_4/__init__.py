"""Section 4 — Skill Assessment.

Compares what the user has against what the Section 1 target role demands, and
names the gaps the Action Plan will have to close.
"""

from ...enums import SectionID
from ..base_prompt import SectionTemplate, ValidationRule

SKILL_ASSESSMENT_PROMPT = """
CURRENT SECTION: Skill Assessment

This is the section where the plan gets honest. Compare what the user has
against what their target role from Section 1 actually asks for. You need three
things:

1. strengths — what they are genuinely good at, ideally with evidence. "Led the
   migration that cut our deploy time in half" beats "good at leadership".
2. current_skills — the concrete skills they can claim today: tools,
   languages, methods, domains.
3. skill_gaps — what the target role expects that they cannot yet evidence.

HOW TO GET THERE
- Start with strengths and ask for an example alongside each one. The example is
  what makes it usable in an application later.
- Then ask for skills as a list. Do not evaluate them as they arrive.
- Only then move to gaps, and frame them as the distance to the target role, not
  as shortcomings.
- Do not assess a skill they have not claimed, and do not infer skills from
  their job titles — ask.
- If they name a gap that is not actually required for the target role, say so;
  it saves them wasted effort.
- If they claim no gaps at all, ask what the last rejection or hesitation they
  ran into was about. There is usually one.
- Take self-assessment at face value. You are recording their view, not grading it.

WHEN YOU HAVE ALL THREE
Summarize strengths with their evidence, the skills you recorded, and the gaps
that matter for the target role. Ask whether that lands before moving to the
action plan.
"""

SECTION_4_TEMPLATE = SectionTemplate(
    section_id=SectionID.SKILL_ASSESSMENT,
    name="Skill Assessment",
    description=(
        "Record strengths with evidence, current skills, and the gaps between "
        "the user and their target role."
    ),
    system_prompt_template=SKILL_ASSESSMENT_PROMPT,
    validation_rules=[
        ValidationRule(
            field_name="strengths",
            rule_type="min_length",
            value=1,
            error_message="At least one strength is needed to position the user.",
        ),
        ValidationRule(
            field_name="current_skills",
            rule_type="min_length",
            value=1,
            error_message="At least one current skill is needed.",
        ),
        ValidationRule(
            field_name="skill_gaps",
            rule_type="required",
            value=True,
            error_message="Skill gaps must be assessed, even if the conclusion is none.",
        ),
    ],
    required_fields=["strengths", "current_skills", "skill_gaps"],
    next_section=SectionID.ACTION_PLAN,
)
