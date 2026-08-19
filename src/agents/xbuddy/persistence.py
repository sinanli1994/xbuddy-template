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
import hashlib
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


# ---------------------------------------------------------------------------
# PR 5 — the final artifact
# ---------------------------------------------------------------------------

# Statuses stored in the durable row. 'current' means the document matches the
# graph's live user_data; 'stale' means source memory moved after it was generated,
# so the content is kept but is no longer authoritative. Invalidation never deletes.
STATUS_CURRENT = "current"
STATUS_STALE = "stale"


def canonical_markdown(text: str | None) -> str:
    """Normalize Markdown just enough that trivia does not read as an edit.

    Line endings are unified and surrounding blank space trimmed, then a single
    trailing newline is restored — which is exactly the shape
    `render_final_output` already produces. Nothing semantic is normalized: this is
    not an equivalence check, and two documents that differ only in wording are
    correctly different.

    A round trip through the editor (Markdown -> Tiptap -> Markdown) can reorder
    whitespace, so it may report an edit where the user changed nothing visible.
    That direction is the safe one: the consequence is refusing to overwrite, never
    discarding someone's work.
    """
    if not text:
        return ""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    return unified.strip() + "\n"


def content_fingerprint(text: str | None) -> str:
    """SHA-256 of the canonical Markdown, hex encoded.

    Deterministic and content-derived on purpose. `updated_at` cannot answer "was
    this edited?" — the agent's own write moves it too — and asking a model would
    make a data-integrity decision non-reproducible.
    """
    return hashlib.sha256(canonical_markdown(text).encode("utf-8")).hexdigest()


def stored_markdown(row: dict[str, Any] | None) -> str | None:
    """The text the editor would display for a row.

    Mirrors the frontend's own `markdown_content || content` preference, so the
    fingerprint is taken over the same bytes the user actually sees and edits.
    """
    if not row:
        return None
    return row.get("markdown_content") or row.get("content")


def is_downstream_edited(row: dict[str, Any] | None) -> bool:
    """Whether a row's content differs from the agent generation it records.

    True means someone edited the document after the agent wrote it, so an
    automatic overwrite would destroy their work.

    A row with no `generated_content_hash` is treated as edited. That is the
    conservative reading: the frontend never writes that column, so its absence
    means no agent generation is on record for this content, and refusing to
    overwrite costs a regeneration while overwriting could cost the user's document.
    """
    if not row:
        return False
    recorded = row.get("generated_content_hash")
    if not recorded:
        return True
    return content_fingerprint(stored_markdown(row)) != recorded


def _fetch_sync(user_id: int, thread_id: str) -> dict[str, Any] | None:
    return _client().get_final_output(user_id=user_id, thread_id=thread_id)


def _save_final_sync(
    user_id: int, thread_id: str, markdown: str, fingerprint: str
) -> dict:
    return _client().save_final_output(
        user_id=user_id,
        thread_id=thread_id,
        content=markdown,
        markdown_content=markdown,
        generated_content_hash=fingerprint,
        status=STATUS_CURRENT,
        agent_id=AGENT_ID,
    )


def _mark_stale_sync(user_id: int, thread_id: str) -> dict:
    return _client().mark_final_output_stale(user_id=user_id, thread_id=thread_id)


async def persist_final_output(
    user_id: int, thread_id: str, markdown: str
) -> tuple[bool, str | None]:
    """Upsert the generated artifact unless doing so would destroy user edits.

    Returns `(persisted, reason)`. Never raises.

    `persisted=False` with a reason is a *refusal*, not necessarily a fault: the
    most important case is a document the user has edited, which must be preserved
    even though that leaves the durable row behind the graph. Preserving edits beats
    automatic synchronization — the graph keeps the new artifact either way.

    Idempotent: the same Markdown produces the same fingerprint, the upsert targets
    the same conflict key, and one logical row results however many times it runs.
    """
    fingerprint = content_fingerprint(markdown)

    try:
        existing = await asyncio.to_thread(_fetch_sync, user_id, thread_id)
    except Exception as exc:  # a read failure must not stall the turn
        logger.exception("persist_final_output: could not read the existing row")
        return False, f"final output read failed: {exc}"

    if existing is not None:
        if existing.get("generated_content_hash") == fingerprint:
            # Byte-identical regeneration of a row we already wrote. Writing again
            # would only move updated_at and wake the editor's realtime listener.
            logger.info("persist_final_output: unchanged; nothing to write")
            return True, None
        if is_downstream_edited(existing):
            logger.warning(
                "persist_final_output: durable content was edited downstream; "
                "refusing to overwrite user_id=%s thread=%s",
                user_id,
                thread_id,
            )
            return False, (
                "final output not written: the stored document has been edited, "
                "so the regenerated version was withheld to preserve those edits"
            )

    try:
        result = await asyncio.to_thread(
            _save_final_sync, user_id, thread_id, markdown, fingerprint
        )
    except Exception as exc:  # credentials or network; never fatal
        logger.exception("persist_final_output: write failed")
        return False, f"final output write failed: {exc}"

    if not result.get("success"):
        return False, f"final output write failed: {result.get('error')}"
    return True, None


async def mark_final_output_stale(user_id: int, thread_id: str) -> tuple[bool, str | None]:
    """Flag the durable row as no longer authoritative, keeping its content.

    Returns `(marked, reason)`. Never raises. Retaining the content is deliberate:
    the user's last document stays readable while the agent rebuilds a replacement,
    and deleting it would lose work in exchange for tidiness.

    A thread with no row is a success with nothing to do — invalidation happens
    whenever source data moves, including before any artifact was ever generated.
    """
    try:
        result = await asyncio.to_thread(_mark_stale_sync, user_id, thread_id)
    except Exception as exc:  # graph invalidation must not depend on this
        logger.exception("mark_final_output_stale: failed")
        return False, f"final output stale marking failed: {exc}"

    if not result.get("success"):
        return False, f"final output stale marking failed: {result.get('error')}"
    return True, None
