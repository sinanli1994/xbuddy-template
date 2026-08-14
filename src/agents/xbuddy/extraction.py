"""Merging extracted section data into the durable XBuddyData record.

Pure functions only — no LLM call, no I/O, no state mutation. The node that
performs the extraction call arrives in Stage 2; this module defines what
"remembering" means once a result is in hand.

The merge is deliberately conservative: extraction can add and correct, but it
can never silently forget. That asymmetry is the whole point — a dropped field
is invisible to the user until the agent re-asks a question they already
answered, which is the exact failure the Stage 7 eval is built to catch.
"""

from typing import Any

from pydantic import BaseModel

from .enums import SectionID
from .models import EXTRACT_MODELS, XBuddyData


def get_extract_model(section_id: SectionID | str) -> type[BaseModel]:
    """Return the extraction schema for a section.

    Raises ValueError for anything that is not one of the five JobBuddy
    sections, matching prompts.get_section_template's contract.
    """
    try:
        key = SectionID(section_id)
    except ValueError as exc:
        raise ValueError(f"Unknown section_id: {section_id!r}") from exc

    try:
        return EXTRACT_MODELS[key]
    except KeyError as exc:  # pragma: no cover - guarded by the registry parity test
        raise ValueError(f"No extraction model registered for {key.value!r}") from exc


def _is_no_op(value: Any) -> bool:
    """Whether an extracted value carries no new information.

    `None` means the model saw nothing new this turn.

    An empty list is treated the same way, and that is a deliberate trade. A
    model that returns `[]` where it meant `null` would otherwise wipe real data,
    and the damage would be silent. The cost is that "explicitly none" cannot be
    recorded — `skill_gaps` is the field where that matters, and expressing it
    needs a separate signal that is out of scope for PR 4.
    """
    return value is None or value == []


def merge_extraction(extracted: BaseModel, user_data: XBuddyData) -> XBuddyData:
    """Apply an extraction result to `user_data`, returning a new instance.

    Semantics, per field:

    * `None`      -> no new information; the stored value is left alone.
    * `[]`        -> same as None (see _is_no_op).
    * any value   -> overwrite. This is how corrections work: a user who says
                     "actually six months" replaces "three months", and a
                     narrowed list replaces the wider one wholesale.

    `user_data` is never mutated — callers rely on being able to compare the
    before and after, and LangGraph state updates must not alias checkpointed
    objects.

    Unknown field names on `extracted` are ignored rather than raising: the
    field-parity test is what pins the schemas together, and a stray key at
    runtime should not break a live turn.
    """
    updates: dict[str, Any] = {}

    for field_name in type(extracted).model_fields:
        if field_name not in XBuddyData.model_fields:
            continue
        value = getattr(extracted, field_name, None)
        if _is_no_op(value):
            continue
        updates[field_name] = value

    if not updates:
        # Nothing to apply. Returning a copy keeps the return type consistent —
        # callers always receive an object they may keep without aliasing.
        return user_data.model_copy(deep=True)

    return user_data.model_copy(update=updates, deep=True)


def extraction_changed(before: XBuddyData, after: XBuddyData) -> bool:
    """Whether a merge actually altered anything.

    Lets the caller emit `user_data` into state only when it changed, matching
    the router's existing "emit only on difference" rule.
    """
    return before != after
