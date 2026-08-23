"""Which Postgres tables LangGraph actually creates, read from the installed library.

A live Fly probe reported `store_vectors` missing and failed a database that was in
fact correctly provisioned. The table was absent because it is *supposed* to be:
`AsyncPostgresStore.setup()` runs its VECTOR_MIGRATIONS only `if self.index_config`,
and JobBuddy constructs `AsyncPostgresStore(pool)` with no index.

Earlier work in this PR got the list wrong in the other direction too — it dropped
`store_migrations`, which `setup()` *does* create unconditionally as its version
ledger, on the strength of reading migration SQL rather than the control flow around
it. Both mistakes came from hardcoding a guess about a third-party library.

So these tests derive the expectation from the **installed** `MIGRATIONS`,
`VECTOR_MIGRATIONS` and `setup()` source rather than restating a list. If a dependency
bump changes what gets created, they fail here — offline, with a clear reason —
instead of on a deployed machine.
"""

import inspect
import re

import pytest
from langgraph.checkpoint.postgres.base import MIGRATIONS as SAVER_MIGRATIONS
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.postgres.base import MIGRATIONS as STORE_MIGRATIONS
from langgraph.store.postgres.base import VECTOR_MIGRATIONS

import memory.postgres as pg

CREATE_TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)")


def tables_in(migrations) -> set[str]:
    """Table names a migration list creates, whatever form its entries take."""
    found: set[str] = set()
    for migration in migrations:
        sql = migration if isinstance(migration, str) else migration.sql
        found.update(CREATE_TABLE.findall(sql))
    return found


# --------------------------------------------------------------------------
# What the installed library actually creates
# --------------------------------------------------------------------------


def test_the_saver_creates_exactly_the_four_checkpoint_tables():
    assert tables_in(SAVER_MIGRATIONS) == {
        "checkpoints",
        "checkpoint_migrations",
        "checkpoint_writes",
        "checkpoint_blobs",
    }


def test_store_vectors_exists_only_in_the_vector_migrations():
    """The whole point. It is not part of the unconditional set."""
    assert "store_vectors" not in tables_in(STORE_MIGRATIONS)
    assert "store_vectors" in tables_in(VECTOR_MIGRATIONS)


def test_the_vector_migrations_are_gated_on_an_index_config():
    """Read from `setup()` itself, so a behaviour change in a future version is caught
    rather than assumed away."""
    source = inspect.getsource(AsyncPostgresStore.setup)
    assert "if self.index_config:" in source
    guard, after = source.split("if self.index_config:", 1)
    assert "VECTOR_MIGRATIONS" in after, "vector migrations must sit behind the guard"
    assert "VECTOR_MIGRATIONS" not in guard, "vector migrations must not run unconditionally"


def test_setup_creates_store_migrations_unconditionally():
    """It is the store's own version ledger, created before any migration runs — which
    is why it does not appear in MIGRATIONS but does appear in the database."""
    source = inspect.getsource(AsyncPostgresStore.setup)
    assert 'table="store_migrations"' in source
    before_guard = source.split("if self.index_config:", 1)[0]
    assert 'table="store_migrations"' in before_guard


def test_the_vector_ledger_is_behind_the_guard_too():
    source = inspect.getsource(AsyncPostgresStore.setup)
    after_guard = source.split("if self.index_config:", 1)[1]
    assert 'table="vector_migrations"' in after_guard


# --------------------------------------------------------------------------
# Our configuration
# --------------------------------------------------------------------------


def test_we_construct_the_store_without_an_index():
    """`AsyncPostgresStore(pool)` — no index argument, so index_config stays None and
    the vector tables are never created."""
    source = inspect.getsource(pg)
    assert "AsyncPostgresStore(self.pool)" in source
    assert "index=" not in source, "an index config here would change the required tables"


@pytest.mark.asyncio
async def test_an_unindexed_store_reports_no_index_config():
    """Async because AsyncPostgresStore's constructor wants a running loop; no
    connection is ever opened."""
    store = AsyncPostgresStore(None)  # type: ignore[arg-type]
    assert store.index_config is None


@pytest.mark.asyncio
async def test_an_indexed_store_would_report_one():
    """Control for the test above — proves `index_config` is not simply always None."""
    store = AsyncPostgresStore(
        None,  # type: ignore[arg-type]
        index={"dims": 8, "embed": lambda texts: [[0.0] * 8 for _ in texts]},
    )
    assert store.index_config is not None


# --------------------------------------------------------------------------
# The permission fallback must require exactly the unconditional set
# --------------------------------------------------------------------------


def required_tables_in_fallback() -> set[str]:
    source = inspect.getsource(pg)
    match = re.search(r"table_name IN \(([^)]*)\)", source)
    assert match, "expected the fallback's table list"
    return set(re.findall(r"'(\w+)'", match.group(1)))


def test_the_fallback_requires_every_unconditionally_created_table():
    expected = tables_in(SAVER_MIGRATIONS) | tables_in(STORE_MIGRATIONS) | {"store_migrations"}
    assert required_tables_in_fallback() == expected


def test_the_fallback_does_not_require_any_optional_vector_table():
    """Requiring these would fail every correctly-provisioned deployment that has no
    index configured — which is all of ours."""
    optional = tables_in(VECTOR_MIGRATIONS) | {"vector_migrations"}
    assert not (required_tables_in_fallback() & optional)


def test_the_fallbacks_count_matches_its_list():
    """The check is `count >= N`; if N and the list drift apart it either passes with
    tables missing or can never pass at all."""
    source = inspect.getsource(pg)
    threshold = re.search(r"if count >= (\d+):", source)
    assert threshold, "expected a count threshold"
    assert int(threshold.group(1)) == len(required_tables_in_fallback())


# --------------------------------------------------------------------------
# The probe must keep the groups separate
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probe_source() -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return (root / "scripts" / "fly_connectivity_probe.py").read_text(encoding="utf-8")


def test_the_probe_separates_saver_store_and_vector_tables(probe_source):
    for name in ("SAVER_TABLES", "STORE_TABLES", "VECTOR_TABLES"):
        assert name in probe_source


def test_the_probe_treats_vector_tables_as_optional(probe_source):
    """They must be reported, not asserted, unless an index is configured.

    The decision has to be *read from the store*. Asserting only that the file
    mentions `index_config` somewhere would still pass for a hardcoded
    `indexed = True`, so the assignment itself is what gets checked.
    """
    assignment = re.search(r"^\s*indexed = (.+)$", probe_source, re.MULTILINE)
    assert assignment, "expected an `indexed = ...` decision"
    assert "index_config" in assignment.group(1), (
        f"the index decision must come from the store, not from {assignment.group(1)!r}"
    )
    assert "no index configured" in probe_source


def test_the_probe_queries_all_three_groups(probe_source):
    """Vector tables must still be *looked up*, even though they are optional.

    Dropping them from the query would make the informational count silently report
    zero present, and would break the check outright if an index were ever configured.
    """
    query_args = re.search(r"\(list\(([^)]*)\),\)", probe_source)
    assert query_args, "expected the table lookup to build its list from the groups"
    for group in ("SAVER_TABLES", "STORE_TABLES", "VECTOR_TABLES"):
        assert group in query_args.group(1), f"{group} missing from the lookup"


def test_the_probes_required_groups_match_the_library(probe_source):
    def literal(name: str) -> set[str]:
        match = re.search(rf"^{name} = \(([^)]*)\)", probe_source, re.MULTILINE)
        assert match, f"expected {name}"
        return set(re.findall(r'"(\w+)"', match.group(1)))

    assert literal("SAVER_TABLES") == tables_in(SAVER_MIGRATIONS)
    assert literal("STORE_TABLES") == tables_in(STORE_MIGRATIONS) | {"store_migrations"}
    assert literal("VECTOR_TABLES") == tables_in(VECTOR_MIGRATIONS) | {"vector_migrations"}


# --------------------------------------------------------------------------
# Does anything actually depend on the store?
# --------------------------------------------------------------------------


def test_no_agent_code_reads_or_writes_the_langgraph_store():
    """Context for how much any of this matters: `agent.store` is assigned in the
    service lifespan and never consumed. The store's tables must exist because
    `setup()` creates them, not because JobBuddy uses them.
    """
    from pathlib import Path

    agents = Path(__file__).resolve().parents[2] / "src" / "agents"
    hits = [
        f"{path.name}:{i}"
        for path in agents.rglob("*.py")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\bstore\.(aput|aget|asearch|adelete|put|get|search)\b", line)
    ]
    assert not hits, f"store usage appeared: {hits}"
