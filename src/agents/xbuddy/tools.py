"""Agent tools.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/tools.py
"""

import logging
from typing import Any

from langchain_core.tools import tool

from .context import build_context_packet
from .enums import SectionStatus
from .models import SectionContent, XBuddyData

logger = logging.getLogger(__name__)


@tool
async def get_context(
    user_id: int,
    thread_id: str,
    section_id: str,
    user_data: dict | None = None,
    status: str = SectionStatus.PENDING.value,
    draft: dict | None = None,
) -> dict:
    """Load the context packet for a section.

    Returns a dict with: section_id, status, system_prompt, draft, validation_rules.

    This is the JSON in/out wrapper over `context.build_context_packet`, for LLM
    and API callers. `router_node` calls the builder directly with typed state
    instead, so the draft does not have to round-trip through JSON.

    `user_id` and `thread_id` are recorded for tracing only. Loading a saved
    draft from Supabase by those keys is PR 4's job — nothing here reads a
    database, so callers must pass `status` and `draft` if they want the packet
    to reflect stored progress.
    """
    logger.debug(
        "get_context: user_id=%s thread_id=%s section_id=%s", user_id, thread_id, section_id
    )

    packet = build_context_packet(
        section_id=section_id,
        status=SectionStatus(status),
        draft=SectionContent.model_validate(draft) if draft else None,
        user_data=XBuddyData.model_validate(user_data) if user_data else None,
    )
    result: dict[str, Any] = packet.model_dump(mode="json")
    return result
