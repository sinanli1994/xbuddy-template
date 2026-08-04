"""Section prompts and navigation helpers.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/prompts.py

`SECTION_TEMPLATES` lives in this module (rather than in sections/) because
service/utils.py:_get_section_name dynamically imports `agents.<name>.prompts`
and reads a module-level `SECTION_TEMPLATES` dict keyed by section-id string.
Matching that convention keeps JobBuddy compatible with the service layer.
"""

from .enums import SectionID
from .sections import ALL_SECTION_TEMPLATES
from .sections.base_prompt import SectionTemplate

# Keyed by SectionID *value* ("career_goal"), matching section_states keys and
# the service layer's lookups.
SECTION_TEMPLATES: dict[str, SectionTemplate] = {
    template.section_id.value: template for template in ALL_SECTION_TEMPLATES
}


def get_section_template(section_id: SectionID | str) -> SectionTemplate:
    """Return the template for a given section.

    Accepts a SectionID or its string value. Raises ValueError for anything
    that is not one of the five JobBuddy sections.
    """
    try:
        key = SectionID(section_id).value
    except ValueError as exc:
        raise ValueError(f"Unknown section_id: {section_id!r}") from exc

    try:
        return SECTION_TEMPLATES[key]
    except KeyError as exc:  # pragma: no cover - guards a missing template file
        raise ValueError(f"No section template registered for {key!r}") from exc


def get_next_section(current: SectionID) -> SectionID | None:
    """Return the next section in sequence, or None if all complete."""
    order = list(SectionID)
    idx = order.index(current)
    if idx + 1 < len(order):
        return order[idx + 1]
    return None


def get_next_unfinished_section(section_states: dict) -> SectionID | None:
    """Find the first section that isn't done yet."""
    for section_id in SectionID:
        state = section_states.get(section_id.value)
        if not state or state.status != "done":
            return section_id
    return None
