from typing import TYPE_CHECKING, Any

from core.logging_config import setup_logging
from core.settings import settings

if TYPE_CHECKING:  # for type checkers only; not imported at runtime
    from core.llm import get_model

__all__ = ["get_model", "settings", "setup_logging"]


def __getattr__(name: str) -> Any:
    """Resolve `get_model` on first use rather than at package import.

    `core.llm` imports eight provider SDKs at module scope — Anthropic, Bedrock,
    Google GenAI, Vertex, Groq, Ollama, OpenAI, and community — and it used to be
    imported eagerly here. Because Python executes a package's `__init__` before any
    submodule, that made `import core.settings` drag in every LLM provider in the
    project, so anything wanting a single configuration value paid for all of them.

    The visible cost was a diagnostic that reads nothing but Postgres settings
    failing with `ModuleNotFoundError: No module named 'langchain_anthropic'` — a
    provider it never uses, reported before it could say anything about the database.
    The invisible cost is that this import chain runs on every cold start.

    PEP 562 module `__getattr__`, so `from core import get_model` still works
    unchanged; only the timing moves. `settings` and `setup_logging` stay eager: they
    are cheap, and `settings` in particular must keep resolving to the Settings
    *instance* that `from core.settings import settings` binds here, not to the
    submodule that a lazy lookup would race with.
    """
    if name == "get_model":
        from core.llm import get_model

        return get_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
