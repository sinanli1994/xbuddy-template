"""Section definitions for the JobBuddy agent.

Single import site for the five section templates. `prompts.SECTION_TEMPLATES`
builds its lookup from these, so section content stays in its own package while
the loader stays in prompts.py (the convention service/utils.py expects).
"""

from .section_1 import SECTION_1_TEMPLATE
from .section_2 import SECTION_2_TEMPLATE
from .section_3 import SECTION_3_TEMPLATE
from .section_4 import SECTION_4_TEMPLATE
from .section_5 import SECTION_5_TEMPLATE

# Ordered to match SectionID declaration order.
ALL_SECTION_TEMPLATES = (
    SECTION_1_TEMPLATE,
    SECTION_2_TEMPLATE,
    SECTION_3_TEMPLATE,
    SECTION_4_TEMPLATE,
    SECTION_5_TEMPLATE,
)

__all__ = [
    "ALL_SECTION_TEMPLATES",
    "SECTION_1_TEMPLATE",
    "SECTION_2_TEMPLATE",
    "SECTION_3_TEMPLATE",
    "SECTION_4_TEMPLATE",
    "SECTION_5_TEMPLATE",
]
