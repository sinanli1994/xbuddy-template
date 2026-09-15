"""Warm expensive, deterministic dependencies before the app accepts requests.

Measured on the Fly machine (shared-cpu-1x) after a restart: the first
`POST /resume/status` took ~13 s. `ResumeStore()` lazily imported `core.llm`, which
then imported every model-provider SDK at module scope — Vertex AI alone pulls in
pandas, numexpr and pyarrow — about 4,000 modules and ~9.7 s on the event loop. The
single worker answered nothing else meanwhile, including Fly's `/health` probe.

`core.llm` now loads provider SDKs on first use, so importing it is cheap. What is
still worth paying before readiness is the SDK the configured model actually uses:
left alone, the first chat turn would import it on the event loop instead. So
startup loads exactly that provider, plus the embedding class when Resume RAG is on
— and nothing for providers that are never selected.

- the model provider is **required**: the graph cannot run without it. A failure here
  fails startup loudly instead of surfacing on the first user request.
- the Supabase client is **optional**: it is built only when Resume RAG is configured,
  and a failure is logged while startup continues. The request path already degrades
  gracefully when Supabase is unavailable.

Local work only. Nothing here constructs or calls a model or an embedding client,
reaches the database, or reads or writes any data.
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
    model_provider: str
    supabase_client: SupabaseWarmup


def _import_provider_stack() -> str:
    """Load the SDK the configured model needs. Returns its provider class name."""
    import core.llm

    return core.llm.preload_model_provider()


def _load_embedding_class() -> None:
    import core.llm

    _ = core.llm.OpenAIEmbeddings  # loads the SDK; constructs nothing


def _construct_supabase_client() -> None:
    from integrations.supabase.supabase_client import get_supabase_client

    get_supabase_client()


def warm_request_dependencies() -> WarmupReport:
    """Load the model provider and build the Supabase client. Idempotent.

    Synchronous and CPU-bound: the lifespan runs it in a worker thread.
    """
    from agents.xbuddy.resume.context import resume_rag_configured

    rag_configured = resume_rag_configured()

    started = time.perf_counter()
    model_provider = _import_provider_stack()
    if rag_configured:
        _load_embedding_class()
    provider_stack_seconds = time.perf_counter() - started

    supabase_client: SupabaseWarmup
    if not rag_configured:
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

    report = WarmupReport(provider_stack_seconds, model_provider, supabase_client)
    logger.info(
        "startup warm-up: model provider %s loaded in %.1fs; Supabase client %s",
        report.model_provider,
        report.provider_stack_seconds,
        report.supabase_client,
    )
    return report
