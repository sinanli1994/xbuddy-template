"""Durable Supabase record for JobBuddy section state.

The agent's only writer of domain data. Deliberately *not* a restore path: the
LangGraph checkpointer remains primary for a live thread, and this layer is the
durable record plus the frontend's read model.

Two boundaries this module holds:

* **Never raises.** `persist_section` returns a bool. A Supabase outage must not
  stall a conversation or make a user re-confirm a summary they already approved.
* **Never blocks the event loop.** `SupabaseClient` is synchronous, so the call is
  offloaded with `asyncio.to_thread`. The existing precedent in the codebase is
  `loop.run_in_executor` (service.py); `to_thread` is the modern equivalent.

Stage 5 wires `persist_section` into `memory_updater_node`, which calls it once
per turn for the current section. Writes that fail are queued in the
`persistence_pending` state key and retried on the next turn, before that turn's
own write — so a failure delays the durable record without ever blocking the
conversation or rolling back a section already marked `DONE`.
"""

import asyncio
import logging
from typing import Any

from .context import _FIELD_LABELS
from .enums import SectionID
from .models import SectionContent, SectionState, XBuddyData
from .prompts import get_section_template

logger = logging.getLogger(__name__)

# Must match what the service's read paths filter on. The column also has a
# 'founder-buddy' default in the migration, which is why every write sends this
# value explicitly rather than relying on the column default.
AGENT_ID = "xbuddy"


def render_section_content(
    section_id: SectionID | str, user_data: XBuddyData
) -> SectionContent:
    """Render a section's collected fields into persistable content.

    Deterministic on purpose. Using the agent's prose summary instead would drift
    between runs and could not be asserted by equality; this can, and it reuses
    the same label map as the `KNOWN SO FAR` block so the user sees consistent
    wording in the app and in the conversation.

    Only the section's own `required_fields` are rendered, and only those with a
    value. An empty section yields `content={}` / `plain_text=""` — `{}` is valid
    JSONB and satisfies the column's NOT NULL, so no migration is needed to
    persist a section that has a status but no data yet.
    """
    template = get_section_template(section_id)

    lines: list[str] = []
    for field_name in template.required_fields:
        value = getattr(user_data, field_name, None)
        if value is None or value == [] or value == "":
            continue
        label = _FIELD_LABELS.get(field_name, field_name.replace("_", " ").capitalize())
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        lines.append(f"{label}: {rendered}")

    if not lines:
        return SectionContent(content={}, plain_text="")

    # Minimal valid Tiptap document — one paragraph per field.
    document: dict[str, Any] = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": line}]}
            for line in lines
        ],
    }
    return SectionContent(content=document, plain_text="\n".join(lines))


def _client():
    """Resolve the Supabase client.

    Indirection exists so tests patch one function instead of reaching into the
    integrations package. Construction can raise when credentials are absent —
    callers treat that as a write failure.
    """
    from integrations.supabase.supabase_client import SupabaseClient

    return SupabaseClient()


def _save_sync(
    user_id: int,
    thread_id: str,
    section_id_value: str,
    section: SectionState,
    content: SectionContent,
) -> dict:
    """The blocking half, run in a worker thread."""
    return _client().save_section_state(
        user_id=user_id,
        thread_id=thread_id,
        section_id=section_id_value,
        content=content.content,
        plain_text=content.plain_text or "",
        status=section.status.value,
        satisfaction_status=section.satisfaction_status,
        agent_id=AGENT_ID,
    )


async def persist_section(
    user_id: int,
    thread_id: str,
    section_id: SectionID | str,
    section: SectionState,
    user_data: XBuddyData,
) -> bool:
    """Upsert one section row. Returns success; never raises.

    Idempotent: the underlying upsert names the
    UNIQUE(user_id, thread_id, section_id) constraint as its conflict target, so
    writing the same turn twice updates one row rather than inserting a duplicate.
    """
    try:
        resolved = SectionID(section_id)
    except ValueError:
        logger.error("persist_section: unknown section_id %r", section_id)
        return False

    content = render_section_content(resolved, user_data)

    try:
        result = await asyncio.to_thread(
            _save_sync, user_id, thread_id, resolved.value, section, content
        )
    except Exception:
        # Covers missing credentials (ValueError from the client factory),
        # network failures, and anything the sync layer lets escape. Logged
        # without the exception text in the return value so a caller cannot
        # accidentally surface connection details.
        logger.exception(
            "persist_section: write failed for section=%s thread=%s",
            resolved.value,
            thread_id,
        )
        return False

    success = bool(isinstance(result, dict) and result.get("success"))
    if not success:
        logger.warning(
            "persist_section: section=%s not saved (%s)",
            resolved.value,
            (result or {}).get("error") if isinstance(result, dict) else "no result",
        )
    return success
