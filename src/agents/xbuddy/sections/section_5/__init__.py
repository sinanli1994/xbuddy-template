"""Section 5 — TODO: rename this section.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/sections/mission/__init__.py

TODO: Define your SectionTemplate here with:
  - section_id: SectionID.ACTION_PLAN
  - name: human-readable name
  - description: what this section covers
  - system_prompt_template: the prompt that guides the LLM in this section
  - validation_rules: what fields are required
  - required_fields: list of field names
  - next_section: None (this is the last section)
"""

from ...enums import SectionID
from ..base_prompt import SectionTemplate

SECTION_5_TEMPLATE = SectionTemplate(
    section_id=SectionID.ACTION_PLAN,
    name="Section 5",
    description="TODO: describe what this section covers",
    system_prompt_template="""
TODO: Write the system prompt for this section.

In this section, you need to gather:
1. ...
2. ...
3. ...

Guidelines:
- Ask one question at a time
- Once you have all elements, present a summary
""",
    validation_rules=[],
    required_fields=[],
    next_section=None,
)
