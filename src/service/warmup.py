"""Warm expensive, deterministic dependencies before the app accepts requests.

Measured on the Fly machine (shared-cpu-1x) after a restart: the first
`POST /resume/status` took ~13 s. `ResumeStore()` lazily imports `core.llm`, and
importing `core.llm` loads every model-provider SDK — Vertex AI alone pulls in
pandas, numexpr and pyarrow — about 4,000 modules and ~9.7 s, all on the event loop.
The single worker answered nothing else meanwhile, including Fly's `/health` probe.

Nothing else on that path was slow. The Supabase client module imported in ~130 ms,
constructed in ~40 ms, and its first request already runs in a worker thread.

So startup pays these once, before the server is declared ready:

- `core.llm` is **required**: the graph cannot run without it. A failure here fails
  startup loudly instead of surfacing on the first user request.
- the Supabase client is **optional**: it is built only when Resume RAG is configured,
  and a failure is logged while startup continues. The request path already degrades
  gracefully when Supabase is unavailable.

Local work only. Nothing here calls a model, an embedding API, or the database, and
nothing is read or written.
"""

import logging
import time
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

SupabaseWarmup = Literal["ready", "not_configured", "failed"]


@dataclass(frozen=True)
class WarmupReport:
    provider_stack_seconds: float
    supabase_client: SupabaseWarmup


def _import_provider_stack() -> None:
    import core.llm  # noqa: F401 — the import is the warm-up


def _construct_supabase_client() -> None:
    from integrations.supabase.supabase_client import get_supabase_client

    get_supabase_client()


def warm_request_dependencies() -> WarmupReport:
    """Import the provider stack and build the Supabase client. Idempotent.

    Synchronous and CPU-bound: the lifespan runs it in a worker thread.
    """
    started = time.perf_counter()
    _import_provider_stack()
    provider_stack_seconds = time.perf_counter() - started

    from agents.xbuddy.resume.context import resume_rag_configured

    supabase_client: SupabaseWarmup
    if not resume_rag_configured():
        supabase_client = "not_configured"
    else:
        try:
            _construct_supabase_client()
            supabase_client = "ready"
        except Exception as exc:  # noqa: BLE001 — optional feature: nothing here may stop startup
            # The type only: a client error can echo configuration back.
            logger.warning(
                "startup warm-up: Supabase client not built (%s); resume features will retry on use",
                type(exc).__name__,
            )
            supabase_client = "failed"

    report = WarmupReport(provider_stack_seconds, supabase_client)
    logger.info(
        "startup warm-up: provider stack imported in %.1fs; Supabase client %s",
        report.provider_stack_seconds,
        report.supabase_client,
    )
    return report
