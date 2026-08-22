"""Deployment diagnostic: does state survive a real machine restart?

Two phases, with a machine restart *between* them. That gap is the entire point. A
single process that closes a pool and reopens it proves reconnection, not
durability — the data could have been in memory the whole time. Only a new process
on a restarted machine tells those apart.

    fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_durability_probe.py write"
    #   -> prints the thread id it created

    fly machine restart <machine-id>

    fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_durability_probe.py read --thread <id>"
    fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_durability_probe.py cleanup --thread <id>"

Use the venv interpreter's absolute path — a bare `python` may resolve to the base
image's interpreter, which has no site-packages.

Model calls are faked, so this costs nothing and needs no OpenAI key, while still
exercising the real graph, the real checkpointer and the real database.

Each phase prints the machine's uptime and pid. If the uptime reported by `read` is
not *lower* than the one reported by `write`, the machine did not actually restart
and the run proves nothing — better to notice that than to assume it.

Secrets are never printed.
"""

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings import DatabaseType, settings

PROBE_USER_ID = 999_003
failures: list[str] = []


def record(label: str, ok: bool, detail: str = "") -> None:
    if not ok:
        failures.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}", flush=True)


def machine_identity() -> str:
    """Uptime and pid, so a claimed restart can be checked rather than trusted."""
    try:
        uptime = f"{float(Path('/proc/uptime').read_text().split()[0]):.0f}s"
    except Exception:  # noqa: BLE001 - not Linux, or /proc unavailable
        uptime = "unknown"
    machine = os.getenv("FLY_MACHINE_ID", "n/a")
    return f"uptime {uptime}, pid {os.getpid()}, FLY_MACHINE_ID={machine}, {sys.executable}"


def sentinel(thread_id: str) -> str:
    return f"durability-probe {thread_id}"


def control_thread(thread_id: str) -> str:
    return f"{thread_id}-control"


def fake_models() -> None:
    """Swap every model call out. Zero cost, no API key, deterministic."""
    from langchain_community.chat_models import FakeListChatModel
    from langchain_core.messages import AIMessage

    from agents.xbuddy import synthesis as synthesis_module
    from agents.xbuddy.enums import DecisionAction
    from agents.xbuddy.models import SectionDecision
    from agents.xbuddy.nodes import generate_decision as decision_module
    from agents.xbuddy.nodes import generate_reply as reply_module
    from agents.xbuddy.nodes import implementation as impl_module
    from agents.xbuddy.nodes import memory_updater as memory_module

    decision = SectionDecision(
        action=DecisionAction.STAY,
        modify_target=None,
        is_satisfied=None,
        user_satisfaction_feedback=None,
        should_save_content=False,
        presented_summary=False,
        decision_reason="durability probe",
    )

    class DecisionChain:
        async def ainvoke(self, messages, config=None):
            return {"raw": AIMessage(content=""), "parsed": decision, "parsing_error": None}

    class ExtractionChain:
        def build(self, model):
            return self

        async def ainvoke(self, messages, config=None):
            return {
                "raw": AIMessage(content=""),
                "parsed": None,
                "parsing_error": ValueError("no-op"),
            }

    async def no_write(*args, **kwargs):
        return True, None

    async def no_section(*args, **kwargs):
        return True

    reply_module._reply_model = lambda: FakeListChatModel(
        responses=["What role are you targeting?"]
    )
    decision_module._decision_chain = lambda: DecisionChain()
    memory_module._extraction_chain = ExtractionChain().build
    memory_module.persist_section = no_section
    memory_module.mark_final_output_stale = no_write
    impl_module.persist_final_output = no_write
    synthesis_module._synthesis_chain = lambda: None


async def connect():
    from memory.postgres import pg_manager

    pg_manager.pool = None
    pg_manager.saver = None
    pg_manager.store = None
    await pg_manager.setup()
    return pg_manager


def build_graph(saver):
    from agents.xbuddy.graph.builder import build_xbuddy_graph

    graph = build_xbuddy_graph()
    graph.checkpointer = saver
    return graph


async def phase_write(thread_id: str) -> None:
    from langchain_core.messages import HumanMessage

    manager = await connect()
    fake_models()
    graph = build_graph(manager.get_saver())
    config = {"configurable": {"thread_id": thread_id, "user_id": PROBE_USER_ID}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content=sentinel(thread_id))]},
        {**config, "recursion_limit": 12},
    )
    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])
    record("checkpoint written", bool(messages), f"{len(messages)} messages")
    record(
        "sentinel present in the written state",
        any(sentinel(thread_id) in str(getattr(m, "content", "")) for m in messages),
    )
    await manager.cleanup()

    read_cmd = (
        "fly ssh console -C "
        f'"/app/.venv/bin/python /app/scripts/fly_durability_probe.py read '
        f'--thread {thread_id}"'
    )
    print(f"\n  thread id: {thread_id}")
    print("  Restart the machine, then run:")
    print(f"    {read_cmd}")


async def phase_read(thread_id: str) -> None:
    manager = await connect()
    fake_models()
    graph = build_graph(manager.get_saver())
    config = {"configurable": {"thread_id": thread_id, "user_id": PROBE_USER_ID}}

    state = await graph.aget_state(config)
    messages = (state.values or {}).get("messages", [])
    record("state restored after restart", bool(messages), f"{len(messages)} messages")
    record(
        "restored state is the SAME state, not a fresh one",
        any(sentinel(thread_id) in str(getattr(m, "content", "")) for m in messages),
        "sentinel matched",
    )
    record("identifiers survived", (state.values or {}).get("user_id") == PROBE_USER_ID)

    other = await graph.aget_state(
        {"configurable": {"thread_id": control_thread(thread_id), "user_id": PROBE_USER_ID}}
    )
    record("a different thread sees nothing", not (other.values or {}))

    await manager.cleanup()


async def phase_cleanup(thread_id: str) -> None:
    manager = await connect()
    threads = [thread_id, control_thread(thread_id)]
    async with manager.pool.connection() as conn:  # type: ignore[union-attr]
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await conn.execute(f"DELETE FROM {table} WHERE thread_id = ANY(%s)", (threads,))
        cursor = await conn.execute(
            "SELECT COUNT(*) AS count FROM checkpoints WHERE thread_id = ANY(%s)", (threads,)
        )
        row = await cursor.fetchone()
    remaining = row["count"] if isinstance(row, dict) else row[0]
    record("probe rows removed", remaining == 0, f"{remaining} left")
    await manager.cleanup()


async def main(phase: str, thread_id: str | None) -> int:
    if settings.DATABASE_TYPE != DatabaseType.POSTGRES:
        print(f"DATABASE_TYPE is {settings.DATABASE_TYPE!s}, not postgres — refusing to run.")
        return 1

    print(f"Phase: {phase}  ({machine_identity()})", flush=True)

    if phase == "write":
        await phase_write(thread_id or f"fly-durability-{uuid.uuid4().hex[:8]}")
    elif not thread_id:
        print("--thread is required here; use the id the write phase printed.")
        return 1
    elif phase == "read":
        await phase_read(thread_id)
    else:
        await phase_cleanup(thread_id)

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["write", "read", "cleanup"])
    parser.add_argument("--thread", help="Thread id from the write phase.")
    args = parser.parse_args()

    if sys.platform == "win32":  # mirrors src/run_service.py
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main(args.phase, args.thread)))
