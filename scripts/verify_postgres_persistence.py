"""Live proof that the deployed Postgres checkpoint path actually persists.

Explicitly opt-in. Nothing under `tests/` imports this, the filename does not match
`test_*.py`, and it lives outside `tests/` — ordinary pytest never collects it. It
opens real connections and writes real rows, which is exactly why it is manual.

    # requires DATABASE_TYPE=postgres and POSTGRES_URI (or the five POSTGRES_* vars)
    uv run python scripts/verify_postgres_persistence.py
    uv run python scripts/verify_postgres_persistence.py --keep   # skip cleanup

What it proves, in order:

    1. the saver initializes against the configured database
    2. the store initializes
    3. a graph run writes a checkpoint
    4. the first pool closes cleanly
    5. a *fresh* pool and saver reconnect, as a redeployed process would
    6. the same thread restores its state across that lifecycle
    7. a different thread does not see that state
    8. the store is reachable and writable

Step 5 is the point. A test that reuses one pool proves nothing about durability —
it could all be in memory. This closes the pool, drops the singleton's references,
and rebuilds, which is the closest a single process gets to a redeploy.

Secrets are never printed. The connection target is reported as host and database
only, never the URI, and never the password.

On Windows this must run on a Selector event loop: psycopg refuses to work async on
the default ProactorEventLoop and raises before it opens a single connection. See
`run_with_compatible_loop` at the bottom, which mirrors what `src/run_service.py`
already does for the real service.
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from core.settings import DatabaseType, settings

VERIFY_USER_ID = 999_002
passed: list[str] = []
failed: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    (passed if condition else failed).append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{f' — {detail}' if detail else ''}")


def describe_target() -> str:
    """Host and database only — never the URI, never the password."""
    from psycopg.conninfo import conninfo_to_dict

    from memory.postgres import pg_manager

    parsed = conninfo_to_dict(pg_manager.get_connection_string())
    host = parsed.get("host", "?")
    port = parsed.get("port", "?")
    dbname = parsed.get("dbname", "?")
    mode = "transaction pooler" if str(port) == "6543" else "direct/session"
    return f"{host}:{port}/{dbname}  ({mode})"


async def fresh_manager():
    """A pool/saver/store built as a newly started process would build them.

    `PostgresConnectionManager` is a singleton, so a genuine second lifecycle means
    clearing its references rather than constructing a second object.
    """
    from memory.postgres import pg_manager

    pg_manager.pool = None
    pg_manager.saver = None
    pg_manager.store = None
    await pg_manager.setup()
    return pg_manager


def build_graph(saver):
    """The real JobBuddy graph, with its checkpointer swapped for the live saver."""
    from agents.xbuddy.graph.builder import build_xbuddy_graph

    graph = build_xbuddy_graph()
    graph.checkpointer = saver
    return graph


def fake_models(monkey: list):
    """Replace every model call so this script costs nothing and needs no API key."""
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
        decision_reason="verification run",
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
    monkey.append("models faked")


async def main(keep: bool) -> int:
    if settings.DATABASE_TYPE != DatabaseType.POSTGRES:
        print(
            f"DATABASE_TYPE is {settings.DATABASE_TYPE!s}, not postgres.\n"
            "Set DATABASE_TYPE=postgres and POSTGRES_URI, then re-run."
        )
        return 1

    from langchain_core.messages import HumanMessage

    thread_a = f"pg-verify-{uuid.uuid4().hex[:8]}"
    thread_b = f"pg-verify-{uuid.uuid4().hex[:8]}"

    try:
        print(f"Target: {describe_target()}")
    except Exception as exc:  # noqa: BLE001 - configuration problem, report and stop
        print(f"  [FAIL] could not resolve the connection target: {type(exc).__name__}")
        return 1
    print(f"Threads: {thread_a} (subject), {thread_b} (control)\n")

    fake_models([])

    # ---------------------------------------------------------- lifecycle 1 ----
    try:
        manager = await fresh_manager()
        check("1. saver initializes", manager.get_saver() is not None)
        check("2. store initializes", manager.get_store() is not None)
    except Exception as exc:  # noqa: BLE001
        check("1. saver initializes", False, f"{type(exc).__name__}: {exc}")
        return 1

    config_a = {"configurable": {"thread_id": thread_a, "user_id": VERIFY_USER_ID}}
    try:
        graph = build_graph(manager.get_saver())
        await graph.ainvoke(
            {"messages": [HumanMessage(content="I need a new job")]},
            {**config_a, "recursion_limit": 12},
        )
        written = await graph.aget_state(config_a)
        check(
            "3. graph writes a checkpoint",
            bool(written.values.get("messages")),
            f"{len(written.values.get('messages', []))} messages",
        )
        message_count = len(written.values.get("messages", []))
    except Exception as exc:  # noqa: BLE001
        check("3. graph writes a checkpoint", False, f"{type(exc).__name__}: {exc}")
        message_count = 0

    # store reachability, on the first lifecycle
    try:
        await manager.get_store().aput(("verify", str(VERIFY_USER_ID)), thread_a, {"ok": True})
        stored = await manager.get_store().aget(("verify", str(VERIFY_USER_ID)), thread_a)
        check("8a. store accepts a write", stored is not None and stored.value == {"ok": True})
    except Exception as exc:  # noqa: BLE001
        check("8a. store accepts a write", False, f"{type(exc).__name__}: {exc}")

    try:
        await manager.cleanup()
        check("4. first pool closes", manager.pool is None)
    except Exception as exc:  # noqa: BLE001
        check("4. first pool closes", False, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------- lifecycle 2 ----
    try:
        manager = await fresh_manager()
        check("5. a fresh pool and saver reconnect", manager.get_saver() is not None)
    except Exception as exc:  # noqa: BLE001
        check("5. a fresh pool and saver reconnect", False, f"{type(exc).__name__}: {exc}")
        return 1

    try:
        graph = build_graph(manager.get_saver())
        restored = await graph.aget_state(config_a)
        restored_messages = (restored.values or {}).get("messages", [])
        check(
            "6. the same thread restores its state",
            len(restored_messages) == message_count and message_count > 0,
            f"{len(restored_messages)} of {message_count} messages",
        )
        check(
            "6b. restored state keeps its identifiers",
            (restored.values or {}).get("user_id") == VERIFY_USER_ID,
        )
    except Exception as exc:  # noqa: BLE001
        check("6. the same thread restores its state", False, f"{type(exc).__name__}: {exc}")

    try:
        config_b = {"configurable": {"thread_id": thread_b, "user_id": VERIFY_USER_ID}}
        other = await graph.aget_state(config_b)
        check("7. a different thread sees nothing", not (other.values or {}))
    except Exception as exc:  # noqa: BLE001
        check("7. a different thread sees nothing", False, f"{type(exc).__name__}: {exc}")

    try:
        survived = await manager.get_store().aget(("verify", str(VERIFY_USER_ID)), thread_a)
        check(
            "8b. store data survives the lifecycle",
            survived is not None and survived.value == {"ok": True},
        )
    except Exception as exc:  # noqa: BLE001
        check("8b. store data survives the lifecycle", False, f"{type(exc).__name__}: {exc}")

    print(
        "\n  note: no JobBuddy code reads the store today — `agent.store` is assigned\n"
        "  but never consumed. 8a/8b prove the store works, not that the product\n"
        "  depends on it."
    )

    # --------------------------------------------------------------- cleanup ----
    if keep:
        print(f"\n--keep: leaving checkpoints for {thread_a} and {thread_b}")
    else:
        try:
            async with manager.pool.connection() as conn:
                for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                    await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = ANY(%s)", ([thread_a, thread_b],)
                    )
            await manager.get_store().adelete(("verify", str(VERIFY_USER_ID)), thread_a)
            async with manager.pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT COUNT(*) AS count FROM checkpoints WHERE thread_id = ANY(%s)",
                    ([thread_a, thread_b],),
                )
                row = await cur.fetchone()
            remaining = row["count"] if isinstance(row, dict) else row[0]
            check("9. cleanup removed the verification rows", remaining == 0, f"{remaining} left")
        except Exception as exc:  # noqa: BLE001
            check("9. cleanup removed the verification rows", False, f"{type(exc).__name__}: {exc}")

    await manager.cleanup()

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    for label in failed:
        print(f"  failed: {label}")
    return 1 if failed else 0


def compatible_loop_policy(platform: str = sys.platform):
    """The event-loop policy psycopg needs here, or None if the default already works.

    psycopg checks this itself on Windows and raises `InterfaceError` the moment a
    connection is attempted on a `ProactorEventLoop`:

        Psycopg cannot use the 'ProactorEventLoop' to run in async mode.

    That happens before any socket, so no amount of credential or SSL fixing helps —
    it has to be settled before the first coroutine runs. `src/run_service.py` already
    sets the same policy for the real service; this keeps the standalone script from
    being the one entrypoint that forgot.
    """
    if platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return None


def run_with_compatible_loop(coro) -> int:
    """Run `coro` on a loop psycopg accepts, leaving POSIX behaviour untouched."""
    policy = compatible_loop_policy()
    if policy is not None:
        asyncio.set_event_loop_policy(policy)
    return asyncio.run(coro)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Leave the verification rows.")
    args = parser.parse_args()
    raise SystemExit(run_with_compatible_loop(main(args.keep)))
