"""Memory updater node — persists section state and manages completion.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/memory_updater.py

This node:
  1. Updates section_states based on the decision
  2. Saves content to Supabase when should_save_content is True
  3. Checks if all sections are done
  4. Sets should_generate_final_output when ready
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..models import XBuddyState

logger = logging.getLogger(__name__)


async def memory_updater_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    """PR 3 compatibility pass-through — deliberately does nothing.

    PR 4 owns everything in this module's docstring: XBuddyData extraction,
    Supabase persistence, PENDING/DONE transitions, and
    should_generate_final_output. Returning an empty update keeps the graph
    traversable so a full conversational turn can complete in PR 3.

    Because it never sets should_generate_final_output, route_after_memory_updater
    always routes back to the router — unchanged from the previous behaviour, and
    the implementation node stays unreachable until PR 5.

    Consequence worth knowing: nothing marks a section DONE yet, so a `next`
    directive cannot actually advance. generate_decision's per-turn reply cap
    bounds that to one extra reply.
    """
    logger.debug("memory_updater: pass-through (PR 4 implements this)")
    empty: dict[str, Any] = {}
    return empty  # type: ignore[return-value]
