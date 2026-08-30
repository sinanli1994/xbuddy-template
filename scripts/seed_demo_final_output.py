"""Seed one throwaway checkpoint carrying a finished career plan.

Explicitly opt-in, and deliberately not a test: it writes real rows to the configured
database. Run it inside the deployed Fly machine, where the Postgres credentials live:

    fly ssh console -C "/app/.venv/bin/python /app/scripts/seed_demo_final_output.py seed"
    fly ssh console -C "/app/.venv/bin/python /app/scripts/seed_demo_final_output.py verify"
    fly ssh console -C "/app/.venv/bin/python /app/scripts/seed_demo_final_output.py cleanup"

Why this exists: `/final_output` had no live proof, because producing a real finished
plan means completing all five sections — many paid model turns. This writes the same
persisted state a finished conversation would leave behind, through the **real
LangGraph checkpointer**, so `/completion` and `/final_output` read it by exactly the
path they read any thread. Nothing is faked at the API layer, nothing is patched in
the frontend, and no model is called.

The identifiers are obviously disposable and the content is synthetic. It never
touches a thread it did not create.

Secrets are never printed.
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings import DatabaseType, settings

# Obviously disposable, and namespaced so cleanup can never match anything else.
SEED_THREAD_ID = "seed-demo-final-output-0001"
SEED_USER_ID = 990_001

PLAN = """# Career Plan

## Career Direction
Target AI Engineer roles focused on LLM applications, RAG systems, AI agents, and
automation.

## Positioning
Highlight production-oriented projects using LangGraph, FastAPI, PostgreSQL, Docker,
cloud deployment, and evaluation tooling.

## Strengths to Leverage
- Production Python and API design
- Distributed systems and database experience
- Comfort with containerised deployment

## Skill Priorities
- AI system design
- RAG evaluation
- Production observability
- Cloud deployment
- Technical interview preparation

## Search Targets
- AI-first product companies
- Platform teams adopting LLM features
- Remote-friendly engineering organisations

## Action Plan
1. Finalise portfolio projects.
2. Update the resume around AI engineering.
3. Practice AI system-design interviews.
4. Apply to targeted AI Engineer roles.

## Still Unknown
- Preferred compensation band
- Willingness to relocate
"""


def seed_state() -> dict:
    """The state a genuinely finished conversation leaves behind.

    Built from the project's own factories and enums rather than hand-written dicts,
    so it matches whatever shape the graph actually persists. Only the fields the two
    read endpoints consult are set to non-default values.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from agents.xbuddy.enums import SectionID, SectionStatus
    from agents.xbuddy.models import SectionState
    from agents.xbuddy.state_factory import build_initial_state

    state = build_initial_state(user_id=SEED_USER_ID, thread_id=SEED_THREAD_ID)
    state["section_states"] = {
        section.value: SectionState(section_id=section, status=SectionStatus.DONE)
        for section in SectionID
    }
    state["current_section"] = SectionID.ACTION_PLAN
    state["should_generate_final_output"] = True
    state["finished"] = True
    state["final_output"] = PLAN
    state["messages"] = [
        HumanMessage(content="I'm exploring AI Engineer roles and want help planning my job search."),
        AIMessage(content="Your job search strategy is ready. You can view it from the sidebar."),
    ]
    return state


async def connect():
    from memory.postgres import pg_manager

    pg_manager.pool = None
    pg_manager.saver = None
    pg_manager.store = None
    await pg_manager.setup()
    return pg_manager


def config() -> dict:
    return {"configurable": {"thread_id": SEED_THREAD_ID, "user_id": SEED_USER_ID}}


async def seed() -> int:
    """Write the checkpoint through the real saver, not by hand-crafting rows."""
    from agents.xbuddy.graph.builder import build_xbuddy_graph

    manager = await connect()
    graph = build_xbuddy_graph()
    graph.checkpointer = manager.get_saver()

    # `aupdate_state` is the checkpointer's own write path, so the row is written
    # exactly as a graph run would write it — no bespoke SQL, no schema assumptions.
    await graph.aupdate_state(config(), seed_state())

    restored = await graph.aget_state(config())
    values = restored.values or {}
    ok = values.get("final_output") == PLAN and values.get("user_id") == SEED_USER_ID
    print(f"  seeded thread : {SEED_THREAD_ID}")
    print(f"  seeded user   : {SEED_USER_ID}")
    print(f"  plan length   : {len(values.get('final_output') or '')} chars")
    print(f"  readback      : {'OK' if ok else 'MISMATCH'}")
    await manager.cleanup()
    return 0 if ok else 1


async def verify() -> int:
    """Read it back through the same helpers the endpoints use."""
    from agents.xbuddy.graph.builder import build_xbuddy_graph
    from service.service import public_completion

    manager = await connect()
    graph = build_xbuddy_graph()
    graph.checkpointer = manager.get_saver()

    values = (await graph.aget_state(config())).values or {}
    completion = public_completion(values)

    print(f"  collection_complete : {completion.collection_complete}")
    print(f"  artifact_available  : {completion.artifact_available}")
    print(f"  sections            : {[(s.id, s.status) for s in completion.sections]}")
    print(f"  final_output chars  : {len(values.get('final_output') or '')}")
    await manager.cleanup()
    return 0 if completion.artifact_available and completion.collection_complete else 1


async def cleanup() -> int:
    """Delete only this seed's rows, matched on the exact thread id."""
    manager = await connect()
    async with manager.pool.connection() as conn:  # type: ignore[union-attr]
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (SEED_THREAD_ID,))
        cursor = await conn.execute(
            "SELECT COUNT(*) AS count FROM checkpoints WHERE thread_id = %s", (SEED_THREAD_ID,)
        )
        row = await cursor.fetchone()
    remaining = row["count"] if isinstance(row, dict) else row[0]
    print(f"  removed rows for {SEED_THREAD_ID}; remaining: {remaining}")
    await manager.cleanup()
    return 0 if remaining == 0 else 1


async def main(action: str) -> int:
    if settings.DATABASE_TYPE != DatabaseType.POSTGRES:
        print(f"DATABASE_TYPE is {settings.DATABASE_TYPE!s}, not postgres — refusing to run.")
        return 1
    return await {"seed": seed, "verify": verify, "cleanup": cleanup}[action]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seed", "verify", "cleanup"])
    args = parser.parse_args()

    if sys.platform == "win32":  # mirrors src/run_service.py
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(args.action)))
