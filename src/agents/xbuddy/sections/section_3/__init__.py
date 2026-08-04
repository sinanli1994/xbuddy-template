"""Section 3 — Job Preferences.

Captures the practical constraints a role has to satisfy, so the Action Plan
targets openings the user would actually accept.
"""

from ...enums import SectionID
from ..base_prompt import SectionTemplate, ValidationRule

JOB_PREFERENCES_PROMPT = """
CURRENT SECTION: Job Preferences

You know what the user wants and where they are coming from. Now establish the
constraints a role has to satisfy for them to say yes. You need five things:

1. preferred_locations — cities, regions, or "remote". More than one is fine.
2. preferred_work_modes — remote, hybrid, on-site, or some combination. If they
   say hybrid, ask how many days on-site they would accept.
3. target_industries — the sectors they want to work in. If they have no
   preference, record that rather than pressing for one.
4. employment_types — full-time, part-time, contract, freelance, internship.
5. salary_expectation — free text, in whatever currency and period they use
   ("90-110k EUR", "about £55k", "day rate around $600").

HOW TO GET THERE
- Ask about location and work mode first; they eliminate the most openings.
- On salary: ask once, plainly, for the range they are targeting. If they would
  rather not say, accept that and record that it is unspecified — do not ask a
  second time and do not guess a number for them.
- If a stated preference rules out most of the target role from Section 1 (a
  fully on-site niche in a small market, say), point that out once and ask
  whether it is a hard constraint or a preference.
- Distinguish "would prefer" from "would decline over". You are looking for
  what actually rules a job out.

WHEN YOU HAVE ALL FIVE
Summarize location, work mode, industries, employment type, and pay
expectation, then ask whether you have it right before moving to their skills.
"""

SECTION_3_TEMPLATE = SectionTemplate(
    section_id=SectionID.JOB_PREFERENCES,
    name="Job Preferences",
    description=(
        "Capture location, work mode, industry, employment type, and pay "
        "expectations that a role must satisfy."
    ),
    system_prompt_template=JOB_PREFERENCES_PROMPT,
    validation_rules=[
        ValidationRule(
            field_name="preferred_locations",
            rule_type="min_length",
            value=1,
            error_message="At least one location (or 'remote') is needed.",
        ),
        ValidationRule(
            field_name="preferred_work_modes",
            rule_type="min_length",
            value=1,
            error_message="At least one work mode is needed to filter openings.",
        ),
        ValidationRule(
            field_name="employment_types",
            rule_type="min_length",
            value=1,
            error_message="At least one employment type is needed.",
        ),
    ],
    required_fields=[
        "preferred_locations",
        "preferred_work_modes",
        "target_industries",
        "employment_types",
        "salary_expectation",
    ],
    next_section=SectionID.SKILL_ASSESSMENT,
)
