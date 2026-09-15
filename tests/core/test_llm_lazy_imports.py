"""core.llm loads a provider SDK only when that provider is used.

It used to import all eight provider SDKs at module scope: ~9.7 s on the Fly machine,
with Vertex AI pulling in pandas, numexpr and pyarrow — paid by every importer,
including callers that only wanted EMBEDDING_DIMENSIONS.

Isolation is tested in subprocesses, because a module already in `sys.modules` cannot
be un-imported in-process. Where providers are "not installed", a meta-path finder
makes them genuinely unimportable. Nothing here makes a network call: models and
embedding clients are constructed with a fake key and never invoked.
"""

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import core.llm
from core.models import (
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    VertexAIModelName,
)

ROOT = Path(__file__).resolve().parents[2]

HEAVY = [
    "pandas",
    "numexpr",
    "pyarrow",
    "vertexai",
    "langchain_google_vertexai",
    "langchain_google_genai",
    "langchain_anthropic",
    "langchain_aws",
    "langchain_groq",
    "langchain_ollama",
    "langchain_community",
]
NON_OPENAI_PROVIDERS = [
    "langchain_anthropic",
    "langchain_aws",
    "langchain_community",
    "langchain_google_genai",
    "langchain_google_vertexai",
    "langchain_groq",
    "langchain_ollama",
]
ALL_PROVIDERS = [*NON_OPENAI_PROVIDERS, "langchain_openai"]

BLOCKER = """
import sys

class Blocked:
    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ModuleNotFoundError(f"No module named {fullname!r}")
        return None

sys.meta_path.insert(0, Blocked(BLOCK_LIST))
"""


def run(body: str, block: list[str] | None = None) -> dict:
    """Run `body` in a fresh interpreter; it must print one JSON object last."""
    script = (
        f"BLOCK_LIST = {block or []!r}\n"
        + BLOCKER
        + f"\nimport json, sys\nsys.path.insert(0, r'{ROOT / 'src'}')\n"
        + f"HEAVY = {HEAVY!r}\n"
        + "def heavy_loaded():\n"
        + "    return sorted({n.split('.')[0] for n in sys.modules} & set(HEAVY))\n"
        + textwrap.dedent(body)
    )
    env = {**os.environ, "OPENAI_API_KEY": "sk-test-not-a-real-key"}
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        capture_output=True, text=True, cwd=ROOT, env=env, check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    assert result.returncode == 0 and lines, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(lines[-1])


# --------------------------------------------------------------------------
# 1. The lightweight path loads no heavy provider
# --------------------------------------------------------------------------


def test_importing_core_llm_loads_no_provider_sdk():
    out = run("""
        import core.llm
        print(json.dumps({"heavy": heavy_loaded(), "openai": "langchain_openai" in sys.modules}))
    """)
    assert out == {"heavy": [], "openai": False}


def test_embedding_constants_import_with_no_provider_installed():
    out = run(
        """
        from core.llm import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
        print(json.dumps({"dims": EMBEDDING_DIMENSIONS, "model": EMBEDDING_MODEL}))
        """,
        block=ALL_PROVIDERS,
    )
    assert out == {"dims": 1536, "model": "text-embedding-3-small"}


def test_the_resume_store_path_loads_no_vertex_pandas_or_numexpr():
    out = run("""
        from agents.xbuddy.resume.store import ResumeStore
        store = ResumeStore()
        print(json.dumps({"dims": store.dimensions, "heavy": heavy_loaded()}))
    """)
    assert out["dims"] == 1536
    assert not {"pandas", "numexpr", "pyarrow", "vertexai", "langchain_google_vertexai"} & set(out["heavy"])


# --------------------------------------------------------------------------
# 2–3. OpenAI still works — with every other provider uninstalled
# --------------------------------------------------------------------------


def test_openai_model_creation_needs_only_the_openai_sdk():
    out = run(
        """
        from core.llm import get_model
        from core.models import OpenAIModelName
        from langchain_openai import ChatOpenAI
        model = get_model(OpenAIModelName.GPT_4O_MINI)
        print(json.dumps({
            "is_chat_openai": isinstance(model, ChatOpenAI),
            "model": model.model_name,
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
            "streaming": model.streaming,
            "heavy": heavy_loaded(),
        }))
        """,
        block=NON_OPENAI_PROVIDERS,
    )
    assert out == {"is_chat_openai": True, "model": "gpt-4o-mini", "temperature": 0.0,
                   "max_tokens": 3000, "streaming": True, "heavy": []}


def test_openai_embeddings_creation_needs_only_the_openai_sdk():
    out = run(
        """
        from core.llm import get_embeddings
        from langchain_openai import OpenAIEmbeddings
        embeddings = get_embeddings()
        print(json.dumps({
            "is_openai_embeddings": isinstance(embeddings, OpenAIEmbeddings),
            "model": embeddings.model,
            "dimensions": embeddings.dimensions,
            "heavy": heavy_loaded(),
        }))
        """,
        block=NON_OPENAI_PROVIDERS,
    )
    assert out == {"is_openai_embeddings": True, "model": "text-embedding-3-small",
                   "dimensions": 1536, "heavy": []}


def test_the_warm_up_loads_the_default_provider_and_nothing_else():
    out = run(
        """
        import agents.xbuddy.resume.context as context
        context.resume_rag_configured = lambda: False  # no Supabase client in this test
        from service.warmup import warm_request_dependencies
        report = warm_request_dependencies()
        print(json.dumps({
            "provider": report.model_provider,
            "openai_loaded": "langchain_openai" in sys.modules,
            "heavy": heavy_loaded(),
        }))
        """,
        block=NON_OPENAI_PROVIDERS,
    )
    assert out == {"provider": "ChatOpenAI", "openai_loaded": True, "heavy": []}


@pytest.mark.parametrize(("rag_on", "expected"), [(True, True), (False, False)])
def test_the_warm_up_preloads_the_embedding_class_only_when_resume_rag_is_on(rag_on, expected):
    """Retrieval and ingestion build OpenAIEmbeddings on the event loop; with Resume
    RAG on, the warm-up resolves the class first. Resolving it caches the class on
    the module, which is what is checked — not merely whether the SDK is imported,
    which the chat model's own preload already guarantees for OpenAI."""
    out = run(
        f"""
        import agents.xbuddy.resume.context as context
        context.resume_rag_configured = lambda: {rag_on}
        import service.warmup as warmup
        warmup._construct_supabase_client = lambda: None  # no real client in this test
        warmup.warm_request_dependencies()
        import core.llm
        print(json.dumps({{"embedding_class_loaded": "OpenAIEmbeddings" in vars(core.llm), "heavy": heavy_loaded()}}))
        """,
        block=NON_OPENAI_PROVIDERS,
    )
    assert out == {"embedding_class_loaded": expected, "heavy": []}


def test_a_non_default_provider_loads_when_selected():
    """Deferred, not removed: asking for Anthropic still loads Anthropic."""
    out = run("""
        import core.llm
        before = "langchain_anthropic" in sys.modules
        core.llm.ChatAnthropic
        print(json.dumps({"before": before, "after": "langchain_anthropic" in sys.modules}))
    """)
    assert out == {"before": False, "after": True}


# --------------------------------------------------------------------------
# 4–5. Public API and provider selection unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(core.llm._PROVIDER_CLASSES))
def test_every_provider_class_is_still_a_module_attribute(name):
    module = importlib.import_module(core.llm._PROVIDER_CLASSES[name])
    assert getattr(core.llm, name) is getattr(module, name)


def test_from_imports_of_provider_classes_still_work():
    from langchain_openai import ChatOpenAI as Real

    from core.llm import ChatOpenAI, OpenAIEmbeddings  # noqa: F401

    assert ChatOpenAI is Real


def test_a_patched_provider_class_is_what_get_model_builds(monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(core.llm, "ChatOpenAI", Recorder)
    model = core.llm.get_model.__wrapped__(OpenAIModelName.GPT_4O)

    assert isinstance(model, Recorder)
    assert model.kwargs == {"model": "gpt-4o", "temperature": 0.0, "max_tokens": 3000, "streaming": True}


def test_fake_tool_model_is_unchanged():
    from langchain_community.chat_models import FakeListChatModel

    fake_tool_model = core.llm.FakeToolModel
    assert issubclass(fake_tool_model, FakeListChatModel)
    assert (fake_tool_model.__name__, fake_tool_model.__module__) == ("FakeToolModel", "core.llm")
    instance = fake_tool_model(responses=["hi"])
    assert instance.bind_tools([object()]) is instance
    assert core.llm.FakeToolModel is fake_tool_model  # defined once


def test_the_model_type_alias_still_resolves():
    from langchain_openai import ChatOpenAI

    assert ChatOpenAI in core.llm.ModelT.__args__


def test_an_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        _ = core.llm.NotAProviderClass


# One member of every family, in get_model's branch order.
FAMILY_SAMPLES = [
    OpenAIModelName.GPT_4O_MINI,
    OpenAICompatibleName.OPENAI_COMPATIBLE,
    AzureOpenAIModelName.AZURE_GPT_4O,
    DeepseekModelName.DEEPSEEK_CHAT,
    AnthropicModelName.HAIKU_3,
    GoogleModelName.GEMINI_20_FLASH,
    VertexAIModelName.GEMINI_20_FLASH,
    GroqModelName.LLAMA_31_8B,
    AWSModelName.BEDROCK_HAIKU,
    OllamaModelName.OLLAMA_GENERIC,
    OpenRouterModelName.GEMINI_25_FLASH,
    FakeModelName.FAKE,
]


@pytest.mark.parametrize("model_name", FAMILY_SAMPLES, ids=str)
def test_the_preload_mapping_names_the_class_get_model_builds(model_name, monkeypatch):
    """The warm-up preloads `model_provider_class_name(...)`; it must be the class
    `get_model` would construct for that model — checked, not assumed."""
    built: list[str] = []

    def sentinel(name):
        class Sentinel:
            def __init__(self, *args, **kwargs):
                built.append(name)

        return Sentinel

    for name in (*core.llm._PROVIDER_CLASSES, "FakeToolModel"):
        monkeypatch.setattr(core.llm, name, sentinel(name))
    for setting, value in {
        "COMPATIBLE_BASE_URL": "http://compatible.test", "COMPATIBLE_MODEL": "compatible-model",
        "AZURE_OPENAI_API_KEY": "azure-test-key", "AZURE_OPENAI_ENDPOINT": "https://azure.test",
        "OLLAMA_MODEL": "llama3.3",
    }.items():
        monkeypatch.setattr(core.llm.settings, setting, value)

    core.llm.get_model.__wrapped__(model_name)

    assert built == [core.llm.model_provider_class_name(model_name)]


def test_the_default_model_preloads_openai():
    assert core.llm.model_provider_class_name() == "ChatOpenAI"
