"""Two Section 1 (Career Goal) prompt variants, differing on one axis.

The axis is **questioning strategy**. Everything else — BASE_RULES, the section's
subject matter, the known-so-far block — is identical, because both variants are
built from `CAREER_GOAL_BODY` through the real `build_system_prompt`. Only a short
strategy block is appended, so any measured difference is attributable to it.

  Variant A (strict)   — one question per turn, nothing else.       ** SELECTED **
  Variant B (anchored) — one primary question plus 2-3 concrete examples.

Variant A won and is what the application now ships: `VARIANT_A_STRATEGY` *is*
`CAREER_GOAL_QUESTIONING_STRATEGY`, imported rather than copied, so this file
cannot drift from the shipped prompt. Variant B is retained unchanged so the
recorded comparison stays reproducible.

The risk being measured: B should be easier to answer, but examples can lead the
user or narrow the field prematurely. The deterministic evaluators could only see
part of that (single_question 0.60); the human rubric caught the rest
(non-leading guidance 2.8 vs 4.6). See README.md for the full result.
"""

from agents.xbuddy.context import build_system_prompt
from agents.xbuddy.enums import SectionID
from agents.xbuddy.models import XBuddyData
from agents.xbuddy.prompts import get_section_template
from agents.xbuddy.sections.section_1 import (
    CAREER_GOAL_BODY,
    CAREER_GOAL_QUESTIONING_STRATEGY,
)

# The shipped strategy, by reference — not a copy.
VARIANT_A_STRATEGY = CAREER_GOAL_QUESTIONING_STRATEGY

VARIANT_B_STRATEGY = """
QUESTIONING STRATEGY
Ask exactly one question, then add two or three concrete example answers so the
user can see the kind of thing you are asking for. Keep the examples short and
visibly non-exhaustive — always leave the door open for something else. The
examples exist to show the shape of a useful answer, never to narrow the field
to what you listed.
"""

VARIANTS: dict[str, str] = {
    "a_strict": VARIANT_A_STRATEGY,
    "b_anchored": VARIANT_B_STRATEGY,
}

# Confirmation arm. Not an A/B variant — it runs the *exact* prompt the
# application ships, to close the one gap between what was measured and what
# shipped: the A/B arms append the strategy block after the KNOWN SO FAR block,
# whereas the shipped template carries it inside system_prompt_template and so
# emits it before. Same content, different order (see build_variant_prompt).
#
# Requested explicitly; it is not part of `--variant all`, which stays as the two
# recorded A/B arms.
SHIPPED_ARM = "shipped"

ALL_ARMS: tuple[str, ...] = (*VARIANTS, SHIPPED_ARM)

# Explicit rather than derived from the arm name. The A/B prefixes match the
# recorded experiments (section1-variant-a-b0abcba7, section1-variant-b-f0b32b99).
EXPERIMENT_PREFIXES: dict[str, str] = {
    "a_strict": "section1-variant-a",
    "b_anchored": "section1-variant-b",
    SHIPPED_ARM: "section1-shipped-confirm",
}


def build_shipped_prompt(user_data: XBuddyData | None = None) -> str:
    """Return the exact prompt the application sends for Career Goal.

    No local composition at all: this is the real section template through the
    real `build_system_prompt`, which is precisely what `router_node` puts into
    `context_packet.system_prompt`. Shared context assembly is untouched, and no
    other section is involved.
    """
    return build_system_prompt(
        get_section_template(SectionID.CAREER_GOAL), user_data or XBuddyData()
    )


def build_variant_prompt(variant: str, user_data: XBuddyData | None = None) -> str:
    """Assemble the full system prompt for an arm.

    For `shipped`, delegates to `build_shipped_prompt` — the exact shipped
    composition, strategy block before KNOWN SO FAR.

    For the two A/B arms, uses `build_system_prompt` over `CAREER_GOAL_BODY` — the
    section's subject matter with no questioning strategy — then appends the arm's
    strategy block, which lands after KNOWN SO FAR. This assembly is deliberately
    unchanged from the recorded run so re-running reproduces the published numbers
    byte for byte; the `shipped` arm exists to cover the ordering difference.
    """
    if variant == SHIPPED_ARM:
        return build_shipped_prompt(user_data)

    if variant not in VARIANTS:
        raise ValueError(f"Unknown arm {variant!r}; expected one of {sorted(ALL_ARMS)}")

    # Strip the shipped strategy back out so neither arm is double-instructed.
    neutral = get_section_template(SectionID.CAREER_GOAL).model_copy(
        update={"system_prompt_template": CAREER_GOAL_BODY}
    )
    base = build_system_prompt(neutral, user_data or XBuddyData())
    return f"{base}\n\n{VARIANTS[variant].strip()}"
