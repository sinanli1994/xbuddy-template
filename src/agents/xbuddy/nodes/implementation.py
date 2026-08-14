"""Implementation node — generates the final output when all sections are complete.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_business_plan.py

In FounderBuddy this generates a business plan. In your XBuddy, this
produces whatever final artifact your agent creates:
  - StudentBuddy: a personalized study plan
  - JobBuddy: a career transition roadmap
  - FitnessBuddy: a structured training program

This node:
  1. Gathers all section data from section_states
  2. Calls the LLM to synthesize a final document
  3. Saves the output to Supabase
  4. Sets finished = True
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..models import XBuddyState

logger = logging.getLogger(__name__)


async def implementation_node(state: XBuddyState, config: RunnableConfig) -> XBuddyState:
    """PR 4 compatibility pass-through — deliberately generates nothing.

    PR 4 Stage 3 is the first thing that can set `should_generate_final_output`,
    and `route_after_memory_updater` sends the turn here the moment it is true
    (graph/routes.py). Until PR 5 exists this node would raise and take the whole
    turn down, so it returns an empty update instead: the turn ends cleanly at
    END with the fifth section correctly marked done.

    PR 5 owns everything in this module's header — gathering section data,
    synthesizing the job search strategy, persisting it, and setting
    `finished = True`. None of that happens here, and no LLM is called.
    """
    logger.debug("implementation: pass-through (PR 5 implements this)")
    empty: dict[str, Any] = {}
    return empty  # type: ignore[return-value]
