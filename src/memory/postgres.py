import logging
from typing import Optional
from urllib.parse import quote

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from core.settings import settings

logger = logging.getLogger(__name__)


class PostgresConnectionManager:
    """Singleton class to manage PostgreSQL connection pool and related components"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.pool: Optional[AsyncConnectionPool] = None
            self.saver: Optional[AsyncPostgresSaver] = None
            self.store: Optional[AsyncPostgresStore] = None
            self.initialized = True
    
    def get_connection_string(self) -> str:
        """The libpq conninfo string the pool connects with.

        `POSTGRES_URI` wins when set and is passed through **verbatim** — including
        its query parameters, so `sslmode`, `options`, and any pooler-specific flag
        survive. Rewriting a URI the operator supplied would be a good way to break
        a connection mode we did not anticipate.

        Otherwise the five discrete fields are assembled as before, which keeps
        existing local setups working unchanged.
        """
        if settings.POSTGRES_URI is not None:
            return settings.POSTGRES_URI.get_secret_value()

        if settings.POSTGRES_PASSWORD is None:
            raise ValueError("POSTGRES_PASSWORD is not set")

        encoded_password = quote(settings.POSTGRES_PASSWORD.get_secret_value(), safe='')

        return (
            f"postgresql://{settings.POSTGRES_USER}:"
            f"{encoded_password}@"
            f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
            f"{settings.POSTGRES_DB}?sslmode=require"
        )
    
    async def setup(self):
        """Initialize connection pool and related components"""
        if self.pool is not None:
            logger.warning("Connection pool already initialized")
            return
        
        validate_postgres_config()
        conn_string = self.get_connection_string()
        
        logger.info("Initializing PostgreSQL connection pool...")

        # Connection pool configuration
        connection_kwargs = {
            "autocommit": True,
            "row_factory": dict_row,
            # Add connection health check parameters
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }

        # Create connection pool - use open=False to avoid deprecation warning
        self.pool = AsyncConnectionPool(
            conninfo=conn_string,
            min_size=2,
            max_size=10,
            timeout=30.0,
            max_lifetime=3600.0,  # 1 hour (increased from 5 minutes)
            max_idle=600.0,  # 10 minutes (increased from 60 seconds)
            kwargs=connection_kwargs,
            open=False,  # Don't open connection pool in constructor
            # Add connection check callback
            check=AsyncConnectionPool.check_connection,
        )
        
        # Explicitly open connection pool
        await self.pool.open()
        
        # Initialize saver and store
        self.saver = AsyncPostgresSaver(self.pool)
        self.store = AsyncPostgresStore(self.pool)

        # Set up database tables
        # Note: Skip setup if tables already exist and user lacks CREATE permission
        try:
            await self.saver.setup()
            await self.store.setup()
            logger.info("Database tables setup completed")
        except Exception as e:
            if "permission denied" in str(e).lower():
                logger.warning("Skipping table setup (user lacks CREATE permission, checking if tables exist)")
                # Verify required tables exist.
                #
                # The four checkpoint_* tables come from the saver's MIGRATIONS. `store`
                # and `store_migrations` come from the store's: `setup()` creates
                # `store_migrations` unconditionally, as its own version ledger.
                #
                # `store_vectors` is deliberately NOT here. It lives in
                # VECTOR_MIGRATIONS, which `AsyncPostgresStore.setup()` runs only
                # `if self.index_config` — and we construct the store as
                # `AsyncPostgresStore(pool)` with no index, so it is never created.
                # Requiring it would fail every correctly-provisioned deployment.
                async with self.pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("""
                            SELECT COUNT(*) as count FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name IN ('checkpoints', 'checkpoint_migrations', 'checkpoint_writes', 'checkpoint_blobs', 'store', 'store_migrations')
                        """)
                        result = await cur.fetchone()
                        count = result['count']
                        if count >= 6:
                            logger.info(f"✓ Verified {count}/6 required tables exist, proceeding without setup")
                        else:
                            raise ValueError(f"Only {count}/6 required tables exist, but cannot create missing tables due to insufficient permissions")
            else:
                raise
        
        logger.info("PostgreSQL connection pool initialized successfully")
    
    async def cleanup(self):
        """Clean up connection pool"""
        if self.pool:
            logger.info("Closing PostgreSQL connection pool...")
            await self.pool.close()
            self.pool = None
            self.saver = None
            self.store = None
            logger.info("PostgreSQL connection pool closed")
    
    def get_saver(self) -> AsyncPostgresSaver:
        """Get saver instance"""
        if self.saver is None:
            raise RuntimeError("Connection pool not initialized. Call setup() first.")
        return self.saver
    
    def get_store(self) -> AsyncPostgresStore:
        """Get store instance"""
        if self.store is None:
            raise RuntimeError("Connection pool not initialized. Call setup() first.")
        return self.store


# Keep original functions for backward compatibility
def validate_postgres_config() -> None:
    """Fail loudly when Postgres is selected but not configured.

    Called from `PostgresConnectionManager.setup()` before any connection is
    attempted, so a misconfigured deployment refuses to start rather than silently
    degrading. There is no fallback to SQLite on this path by design: quietly
    switching a deployed service to an ephemeral file would lose every
    conversation on the next restart, and would do it invisibly.

    A single `POSTGRES_URI` satisfies the requirement on its own.
    """
    if settings.POSTGRES_URI is not None:
        return

    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD", 
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]
    
    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "Set POSTGRES_URI, or set all of these environment variables, to use "
            "PostgreSQL persistence."
        )


def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from settings."""
    # Use connection manager's method
    return pg_manager.get_connection_string()


# Global connection manager instance
pg_manager = PostgresConnectionManager()


def get_postgres_saver():
    """Get PostgreSQL saver (maintain backward compatibility)"""
    return pg_manager.get_saver()


def get_postgres_store():
    """Get PostgreSQL store (maintain backward compatibility)"""
    return pg_manager.get_store()