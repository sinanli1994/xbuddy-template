"""Deployment diagnostic: can this machine actually reach Postgres?

Run it inside the deployed Fly machine, by hand:

    fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_connectivity_probe.py"

Use the venv interpreter's absolute path. A bare `python` depends on the image's
PATH reaching the ssh session, and if it does not you get the base image's
interpreter with no site-packages — which fails on the first third-party import
and looks like a missing dependency rather than a wrong interpreter.

Not an endpoint, not wired into the app, and it takes no arguments that could
change behaviour. It answers one question — whether the configured database is
reachable and usable from *this* network position — and separates that from every
downstream question about durability.

It exists because local Windows validation could not complete a TCP handshake to
the Supabase pooler on 5432 (port 443 to the same host was fine), so authentication,
TLS and role behaviour were never exercised. Fly's network path is independent, and
this is the first place those can honestly be tested.

Secrets are never printed. The target is reported as host/port/database; the URI and
password are never emitted, not even on the failure paths.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings import DatabaseType, settings

# Created unconditionally by AsyncPostgresSaver.setup() (its MIGRATIONS).
SAVER_TABLES = ("checkpoints", "checkpoint_migrations", "checkpoint_writes", "checkpoint_blobs")

# Created unconditionally by AsyncPostgresStore.setup(): `store` from its MIGRATIONS,
# `store_migrations` from the version ledger `setup()` creates before running them.
STORE_TABLES = ("store", "store_migrations")

# Created ONLY when the store is constructed with an index config —
# `setup()` guards VECTOR_MIGRATIONS behind `if self.index_config`. Optional.
VECTOR_TABLES = ("store_vectors", "vector_migrations")

steps: list[tuple[str, bool, str]] = []


def record(label: str, ok: bool, detail: str = "") -> None:
    steps.append((label, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}", flush=True)


def sanitized_target() -> str:
    """Host, port and database name only. Never the URI, never the password."""
    from psycopg.conninfo import conninfo_to_dict

    from memory.postgres import pg_manager

    parsed = conninfo_to_dict(pg_manager.get_connection_string())
    port = str(parsed.get("port", "?"))
    mode = {"6543": "transaction pooler", "5432": "session pooler / direct"}.get(port, "unknown")
    return f"{parsed.get('host', '?')}:{port}/{parsed.get('dbname', '?')}  ({mode})"


async def probe() -> int:
    # Which interpreter this is, so a PATH problem is never mistaken for a missing
    # dependency again. Everything below depends on it being the project venv.
    print(f"Interpreter: {sys.executable}", flush=True)

    if settings.DATABASE_TYPE != DatabaseType.POSTGRES:
        print(f"DATABASE_TYPE is {settings.DATABASE_TYPE!s}, not postgres — nothing to probe.")
        return 1

    # 1. configuration resolves at all
    try:
        print(f"Target: {sanitized_target()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] configuration did not resolve: {type(exc).__name__}: {exc}")
        return 1

    from memory.postgres import pg_manager

    # 2. a bare connection, before any pool or LangGraph machinery is involved.
    #    If this fails the problem is network/credentials, not our code.
    try:
        import psycopg

        conn = await psycopg.AsyncConnection.connect(
            pg_manager.get_connection_string(), connect_timeout=15
        )
        try:
            cursor = await conn.execute("SELECT 1")
            row = await cursor.fetchone()
            record("direct psycopg connection + SELECT 1", row is not None and row[0] == 1)
            cursor = await conn.execute("SELECT current_user, current_database(), version()")
            user, database, version = await cursor.fetchone()  # type: ignore[misc]
            record("server identifies itself", True, f"{user}@{database} — {version.split(',')[0]}")
        finally:
            await conn.close()
    except Exception as exc:  # noqa: BLE001
        record("direct psycopg connection + SELECT 1", False, f"{type(exc).__name__}: {exc}")
        print(
            "\n  A failure here is network, TLS or credentials — reached before any "
            "application logic.\n  Nothing below can succeed, so stopping.",
            flush=True,
        )
        return 1

    # 3. the same pool path the service uses, including saver/store DDL
    try:
        pg_manager.pool = None
        pg_manager.saver = None
        pg_manager.store = None
        await pg_manager.setup()
        record("pg_manager.setup() — pool opened", pg_manager.pool is not None)
        record("AsyncPostgresSaver ready", pg_manager.get_saver() is not None)
        record("AsyncPostgresStore ready", pg_manager.get_store() is not None)
    except Exception as exc:  # noqa: BLE001
        record("pg_manager.setup()", False, f"{type(exc).__name__}: {exc}")
        return 1

    # 4. the tables setup() is responsible for creating
    #
    # Checked as three separate groups, because they are created by three different
    # mechanisms and only two of them are unconditional. Lumping them into one count
    # is what produced a false failure: `store_vectors` was reported missing on a
    # database that was in fact correctly provisioned.
    try:
        async with pg_manager.pool.connection() as conn:  # type: ignore[union-attr]
            cursor = await conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY(%s)",
                (list(SAVER_TABLES + STORE_TABLES + VECTOR_TABLES),),
            )
            rows = await cursor.fetchall()
        present = {r["table_name"] if isinstance(r, dict) else r[0] for r in rows}

        for label, expected in (
            ("saver tables", SAVER_TABLES),
            ("store tables", STORE_TABLES),
        ):
            missing = [t for t in expected if t not in present]
            record(
                f"{label} present",
                not missing,
                f"{len(expected) - len(missing)}/{len(expected)}"
                + (f", missing: {', '.join(missing)}" if missing else ""),
            )

        # Vector tables are created only when the store is built with an index
        # config. We build `AsyncPostgresStore(pool)` with none, so their absence is
        # the correct outcome — reported for information, never as a failure.
        indexed = getattr(pg_manager.get_store(), "index_config", None) is not None
        found_vector = [t for t in VECTOR_TABLES if t in present]
        if indexed:
            record(
                "vector tables present (index configured)",
                len(found_vector) == len(VECTOR_TABLES),
                f"{len(found_vector)}/{len(VECTOR_TABLES)}",
            )
        else:
            print(
                f"  [n/a ] vector tables — no index configured, so LangGraph never "
                f"creates them ({len(found_vector)} present)",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        record("LangGraph tables present", False, f"{type(exc).__name__}: {exc}")

    # 5. close cleanly — a pool that will not shut down is a deployment problem
    try:
        await pg_manager.cleanup()
        record("pool closed cleanly", pg_manager.pool is None)
    except Exception as exc:  # noqa: BLE001
        record("pool closed cleanly", False, f"{type(exc).__name__}: {exc}")

    failures = [label for label, ok, _ in steps if not ok]
    print(f"\n{len(steps) - len(failures)} passed, {len(failures)} failed", flush=True)
    return 1 if failures else 0


def run() -> int:
    # Matches src/run_service.py. A no-op on the Linux machine this is built for,
    # but psycopg refuses to work on Windows' default ProactorEventLoop, so the
    # guard keeps the script usable from a developer laptop too.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(probe())


if __name__ == "__main__":
    raise SystemExit(run())
