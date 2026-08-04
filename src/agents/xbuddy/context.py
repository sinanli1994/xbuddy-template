"""Assembly of the per-section context packet.

`router_node` calls `build_context_packet` directly with typed state; the
`get_context` tool in tools.py wraps the same functions with JSON in/out. Both
paths therefore produce identical content — the same split PR 1 used for
state_factory.py.

Nothing here calls an LLM. Prompt assembly is pure string work over the section
template and the data already collected in state.
"""

from typing import Any

from .enums import SectionID, SectionStatus
from .models import ContextPacket, SectionContent, XBuddyData
from .prompts import get_section_template
from .sections.base_prompt import BASE_RULES, SectionTemplate

# Rendered labels for XBuddyData fields, so the "known so far" block reads as
# prose rather than as attribute names.
_FIELD_LABELS: dict[str, str] = {
    "target_roles": "Target role(s)",
    "career_goal_summary": "Career goal",
    "target_timeline": "Timeline",
    "current_role": "Current role",
    "years_experience": "Years of experience",
    "highest_education": "Education",
    "work_history": "Work history",
    "preferred_locations": "Preferred locations",
    "preferred_work_modes": "Work modes",
    "target_industries": "Target industries",
    "employment_types": "Employment types",
    "salary_expectation": "Salary expectation",
    "strengths": "Strengths",
    "current_skills": "Current skills",
    "skill_gaps": "Skill gaps",
    "action_items": "Action items",
}


def render_known_data(user_data: XBuddyData) -> str:
    """Render the non-empty fields of `user_data` as a plain-text block.

    Only filled fields appear. An empty list or None is omitted entirely rather
    than rendered as a placeholder, which is what keeps the agent's
    no-placeholder rule enforceable: if a field is missing from this block, the
    agent has to ask for it.

    Returns an empty string when nothing has been collected yet.
    """
    lines: list[str] = []
    for field_name in user_data.__class__.model_fields:
        value = getattr(user_data, field_name, None)
        if value is None or value == [] or value == "":
            continue
        label = _FIELD_LABELS.get(field_name, field_name.replace("_", " ").capitalize())
        if isinstance(value, list):
            lines.append(f"- {label}: {', '.join(str(item) for item in value)}")
        else:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def build_system_prompt(template: SectionTemplate, user_data: XBuddyData | None = None) -> str:
    """Compose BASE_RULES + the section's own prompt + what is already known.

    The section template is inserted literally — never through `str.format()` —
    because prompts contain literal braces (JSON and Tiptap examples) that would
    raise KeyError/IndexError.
    """
    parts = [BASE_RULES.strip(), template.system_prompt_template.strip()]

    known = render_known_data(user_data) if user_data is not None else ""
    if known:
        parts.append(f"KNOWN SO FAR\n{known}")
    else:
        parts.append("KNOWN SO FAR\nNothing collected yet — this is the start of the conversation.")

    return "\n\n".join(parts)


def build_validation_rules(template: SectionTemplate) -> dict[str, Any]:
    """Flatten a template's rules into ContextPacket's dict shape.

    SectionTemplate.validation_rules is a list[ValidationRule] while
    ContextPacket.validation_rules is dict[str, Any] | None, so the two need
    bridging rather than one of the models changing.
    """
    return {
        "required_fields": list(template.required_fields),
        "rules": [rule.model_dump() for rule in template.validation_rules],
    }


def build_context_packet(
    section_id: SectionID | str,
    status: SectionStatus = SectionStatus.PENDING,
    draft: SectionContent | None = None,
    user_data: XBuddyData | None = None,
) -> ContextPacket:
    """Build the ContextPacket the reply node consumes for a section."""
    template = get_section_template(section_id)
    return ContextPacket(
        section_id=template.section_id,
        status=SectionStatus(status),
        system_prompt=build_system_prompt(template, user_data),
        draft=draft,
        validation_rules=build_validation_rules(template),
    )
