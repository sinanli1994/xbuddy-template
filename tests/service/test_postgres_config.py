"""PR 6 Stage 5A: the Postgres runtime contract, proved offline.

Nothing here opens a socket. It pins which settings the production path actually
reads, that a single URI is sufficient, that the branch is selected only by
`DATABASE_TYPE`, and — most importantly — that a misconfigured Postgres deployment
**fails loudly instead of silently becoming ephemeral SQLite**.

The live durability proof is `scripts/verify_postgres_persistence.py`, which is
opt-in and never runs from pytest.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import SecretStr

import memory
import memory.postgres as pg
from core.settings import DatabaseType, Settings

SUPABASE_STYLE_URI = (
    "postgresql://postgres.abcdefghijkl:p%40ss%2Fword"
    "@aws-0-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require"
)

BASE = {"OPENAI_API_KEY": SecretStr("sk-test"), "_env_file": None}


def uri_settings(**overrides) -> Settings:
    return Settings(POSTGRES_URI=SecretStr(SUPABASE_STYLE_URI), **BASE, **overrides)


def discrete_settings(**overrides) -> Settings:
    """The five-field form. `overrides` may blank any field, so they are merged
    into the defaults rather than passed alongside them (which duplicates keys)."""
    fields = {
        "POSTGRES_USER": "jobbuddy",
        "POSTGRES_PASSWORD": SecretStr("p@ss/word"),
        "POSTGRES_HOST": "db.example.supabase.co",
        "POSTGRES_PORT": 5432,
        "POSTGRES_DB": "postgres",
    }
    fields.update(overrides)
    return Settings(**fields, **BASE)


# --------------------------------------------------------------------------
# Which settings the production path actually reads
# --------------------------------------------------------------------------


def test_supabase_db_url_is_not_consumed_by_the_runtime():
    """It is declared on Settings but read nowhere.

    Recorded as a test so nobody configures it expecting persistence to work.
    `POSTGRES_URI` is the field the pool actually uses.
    """
    settings = Settings(SUPABASE_DB_URL="postgresql://ignored/db", **BASE)
    with patch.object(pg, "settings", settings), pytest.raises(ValueError, match="Missing required"):
        pg.validate_postgres_config()


def test_a_single_uri_is_sufficient_configuration():
    with patch.object(pg, "settings", uri_settings()):
        pg.validate_postgres_config()  # must not raise


def test_the_uri_is_passed_through_verbatim():
    """Query parameters carry sslmode and any pooler flag; rewriting them would
    silently change the connection mode."""
    with patch.object(pg, "settings", uri_settings()):
        assert pg.pg_manager.get_connection_string() == SUPABASE_STYLE_URI


def test_the_uri_wins_over_the_discrete_fields():
    settings = discrete_settings(POSTGRES_URI=SecretStr(SUPABASE_STYLE_URI))
    with patch.object(pg, "settings", settings):
        assert pg.pg_manager.get_connection_string() == SUPABASE_STYLE_URI


def test_the_discrete_fields_still_work_unchanged():
    """The local path must not regress just because a URI form was added."""
    with patch.object(pg, "settings", discrete_settings()):
        pg.validate_postgres_config()
        built = pg.pg_manager.get_connection_string()

    assert built.startswith("postgresql://jobbuddy:")
    assert "@db.example.supabase.co:5432/postgres" in built
    assert built.endswith("?sslmode=require")


def test_the_discrete_password_is_percent_encoded():
    """`p@ss/word` would otherwise corrupt the URI's authority section."""
    with patch.object(pg, "settings", discrete_settings()):
        built = pg.pg_manager.get_connection_string()
    assert "p%40ss%2Fword" in built
    assert "p@ss/word" not in built.split("@db.example")[0].removeprefix(
        "postgresql://jobbuddy:"
    )


def test_the_uri_is_held_as_a_secret():
    """It embeds the password, so it must not appear in a repr or a log line."""
    settings = uri_settings()
    assert isinstance(settings.POSTGRES_URI, SecretStr)
    assert "pooler.supabase.com" not in repr(settings.POSTGRES_URI)
    assert "p%40ss" not in str(settings.POSTGRES_URI)


# --------------------------------------------------------------------------
# Startup failure semantics: no silent fallback
# --------------------------------------------------------------------------


def test_unconfigured_postgres_raises_rather_than_defaulting():
    with patch.object(pg, "settings", Settings(**BASE)), pytest.raises(
        ValueError, match="Missing required PostgreSQL configuration"
    ):
        pg.validate_postgres_config()


def test_the_error_names_both_configuration_forms():
    with patch.object(pg, "settings", Settings(**BASE)):
        with pytest.raises(ValueError) as excinfo:
            pg.validate_postgres_config()
    message = str(excinfo.value)
    assert "POSTGRES_URI" in message
    assert "POSTGRES_USER" in message


@pytest.mark.parametrize(
    "missing",
    ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"],
)
def test_a_partial_discrete_configuration_is_rejected(missing):
    settings = discrete_settings(**{missing: None})
    with patch.object(pg, "settings", settings), pytest.raises(ValueError, match=missing):
        pg.validate_postgres_config()


@pytest.mark.asyncio
async def test_setup_validates_before_opening_a_connection():
    """The failure must happen before any socket is attempted."""
    manager = pg.PostgresConnectionManager()
    manager.pool = None
    with patch.object(pg, "settings", Settings(**BASE)):
        with patch.object(pg, "AsyncConnectionPool") as pool_cls:
            with pytest.raises(ValueError, match="Missing required"):
                await manager.setup()
    pool_cls.assert_not_called()


@pytest.mark.asyncio
async def test_the_postgres_branch_never_falls_back_to_sqlite(monkeypatch):
    """A deployed service must not quietly become ephemeral.

    Falling back would lose every conversation on the next restart, invisibly.
    """
    settings = Settings(DATABASE_TYPE=DatabaseType.POSTGRES, **BASE)
    monkeypatch.setattr(memory, "settings", settings)
    monkeypatch.setattr(pg, "settings", settings)
    pg.pg_manager.pool = None

    sqlite_used = Mock()
    monkeypatch.setattr(memory, "get_sqlite_saver", sqlite_used)

    with pytest.raises(ValueError, match="Missing required PostgreSQL configuration"):
        async with memory.initialize_database():
            pass

    sqlite_used.assert_not_called()


@pytest.mark.asyncio
async def test_the_store_branch_never_falls_back_either(monkeypatch):
    settings = Settings(DATABASE_TYPE=DatabaseType.POSTGRES, **BASE)
    monkeypatch.setattr(memory, "settings", settings)
    monkeypatch.setattr(pg, "settings", settings)
    pg.pg_manager.pool = None

    sqlite_store = Mock()
    monkeypatch.setattr(memory, "get_sqlite_store", sqlite_store)

    with pytest.raises(ValueError):
        async with memory.initialize_store():
            pass

    sqlite_store.assert_not_called()


# --------------------------------------------------------------------------
# Branch selection
# --------------------------------------------------------------------------


def test_sqlite_is_the_default_database_type():
    assert Settings(**BASE).DATABASE_TYPE is DatabaseType.SQLITE


@pytest.mark.asyncio
async def test_only_database_type_postgres_selects_the_postgres_branch(monkeypatch):
    """sqlite and mongo must not touch pg_manager."""
    for database_type in (DatabaseType.SQLITE, DatabaseType.MONGO):
        settings = Settings(DATABASE_TYPE=database_type, **BASE)
        monkeypatch.setattr(memory, "settings", settings)
        touched = Mock()
        monkeypatch.setattr(pg.pg_manager, "setup", touched)

        saver = Mock()

        class Ctx:
            async def __aenter__(self):
                return saver

            async def __aexit__(self, *args):
                return False

        monkeypatch.setattr(memory, "get_sqlite_saver", lambda: Ctx())
        monkeypatch.setattr(memory, "get_mongo_saver", lambda: Ctx())

        async with memory.initialize_database() as resolved:
            assert resolved is saver
        touched.assert_not_called()


@pytest.mark.asyncio
async def test_the_postgres_branch_yields_the_pooled_saver_and_store(monkeypatch):
    settings = Settings(DATABASE_TYPE=DatabaseType.POSTGRES, **BASE)
    monkeypatch.setattr(memory, "settings", settings)

    saver, store = Mock(name="saver"), Mock(name="store")
    monkeypatch.setattr(pg.pg_manager, "pool", Mock(name="pool"))
    monkeypatch.setattr(pg.pg_manager, "get_saver", Mock(return_value=saver))
    monkeypatch.setattr(pg.pg_manager, "get_store", Mock(return_value=store))
    monkeypatch.setattr(pg.pg_manager, "setup", AsyncMock())

    async with memory.initialize_database() as resolved_saver:
        assert resolved_saver is saver
    async with memory.initialize_store() as resolved_store:
        assert resolved_store is store


def test_the_local_sqlite_path_is_untouched():
    """SQLite still uses the configured file and an in-memory store."""
    import memory.sqlite as sq

    source = __import__("inspect").getsource(sq)
    assert "settings.SQLITE_DB_PATH" in source
    assert "InMemoryStore" in source


# --------------------------------------------------------------------------
# What setup() actually configures
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_pool_receives_the_conninfo_and_session_kwargs():
    """Records the configuration the saver/store are handed.

    `autocommit` and the `keepalives_*` family are **session-level** settings, which
    is the concrete reason a transaction-mode pooler is the wrong target.
    """
    manager = pg.PostgresConnectionManager()
    manager.pool = None
    captured = {}

    class FakePool:
        # `setup()` passes `check=AsyncConnectionPool.check_connection`, and that
        # attribute is looked up on whatever the name is bound to — so the stand-in
        # has to provide it too.
        check_connection = staticmethod(lambda conn: None)

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def open(self):
            return None

    with patch.object(pg, "settings", uri_settings()):
        with patch.object(pg, "AsyncConnectionPool", FakePool):
            with patch.object(pg, "AsyncPostgresSaver") as saver_cls:
                with patch.object(pg, "AsyncPostgresStore") as store_cls:
                    saver_cls.return_value.setup = AsyncMock()
                    store_cls.return_value.setup = AsyncMock()
                    await manager.setup()

    assert captured["conninfo"] == SUPABASE_STYLE_URI
    assert captured["kwargs"]["autocommit"] is True
    assert captured["kwargs"]["keepalives"] == 1
    assert captured["min_size"] == 2 and captured["max_size"] == 10

    # Both components share the one pool.
    saver_cls.assert_called_once()
    store_cls.assert_called_once()
    assert saver_cls.call_args.args[0] is store_cls.call_args.args[0]

    manager.pool = None
    manager.saver = None
    manager.store = None


def test_the_permission_fallback_requires_only_unconditional_tables():
    """`store_vectors` must NOT be required.

    An earlier revision of this test asserted the opposite, on the strength of reading
    migration SQL rather than the control flow around it. A live probe then failed a
    correctly-provisioned database for a missing `store_vectors` — a table
    `AsyncPostgresStore.setup()` creates only `if self.index_config`, which is never
    set here. `store_migrations` is the one that really is unconditional.

    The expectation is derived from the installed library in
    `test_store_table_expectations.py`; this keeps the specific regression named where
    the fallback itself is tested.
    """
    import inspect
    import re

    # Parse the list rather than searching the file: the comment above it explains
    # why `store_vectors` is excluded, so a substring search matches its own
    # documentation and passes — or fails — for the wrong reason.
    source = inspect.getsource(pg)
    match = re.search(r"table_name IN \(([^)]*)\)", source)
    assert match, "expected the fallback's table list"
    required = set(re.findall(r"'(\w+)'", match.group(1)))

    assert "store_migrations" in required
    assert "store_vectors" not in required
    assert required == {
        "checkpoints",
        "checkpoint_migrations",
        "checkpoint_writes",
        "checkpoint_blobs",
        "store",
        "store_migrations",
    }


def test_getters_raise_before_setup():
    manager = pg.PostgresConnectionManager()
    saved_saver, saved_store = manager.saver, manager.store
    manager.saver = None
    manager.store = None
    try:
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_saver()
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_store()
    finally:
        manager.saver, manager.store = saved_saver, saved_store
