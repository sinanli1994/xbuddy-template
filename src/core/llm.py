import sys
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

try:
    from typing import TypeAlias
except ImportError:
    # Python 3.9 compatibility
    TypeAlias = str

from core.models import (
    AllModelEnum,
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
from core.settings import settings

if TYPE_CHECKING:  # the real classes, for type checkers only — see _PROVIDER_CLASSES
    from langchain_anthropic import ChatAnthropic
    from langchain_aws import ChatBedrock
    from langchain_community.chat_models import FakeListChatModel
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_vertexai import ChatVertexAI
    from langchain_groq import ChatGroq
    from langchain_ollama import ChatOllama
    from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings

    class FakeToolModel(FakeListChatModel):
        def __init__(self, responses: list[str]): ...

        def bind_tools(self, tools): ...

    ModelT: TypeAlias = (
        AzureChatOpenAI
        | ChatOpenAI
        | ChatAnthropic
        | ChatGoogleGenerativeAI
        | ChatVertexAI
        | ChatGroq
        | ChatBedrock
        | ChatOllama
        | FakeToolModel
    )

# =============================================================================
# Provider SDKs load on first use
# =============================================================================
#
# This module used to import all eight provider SDKs at module scope. On the Fly
# machine that cost ~9.7 s — Vertex AI alone drags in pandas, numexpr and pyarrow —
# and every importer paid it, including callers that wanted nothing but
# EMBEDDING_DIMENSIONS. Now a provider's SDK is imported only when a model from that
# provider is built, or when its class is named as `core.llm.<Class>`.
#
# The class names stay module attributes (PEP 562), so `core.llm.ChatOpenAI`,
# `from core.llm import ChatOpenAI` and `patch("core.llm.ChatOpenAI")` all behave as
# before. `get_model` looks classes up through the module for the same reason: a
# patched attribute is what it builds.

_PROVIDER_CLASSES: dict[str, str] = {
    "AzureChatOpenAI": "langchain_openai",
    "ChatOpenAI": "langchain_openai",
    "OpenAIEmbeddings": "langchain_openai",
    "ChatAnthropic": "langchain_anthropic",
    "ChatBedrock": "langchain_aws",
    "FakeListChatModel": "langchain_community.chat_models",
    "ChatGoogleGenerativeAI": "langchain_google_genai",
    "ChatVertexAI": "langchain_google_vertexai",
    "ChatGroq": "langchain_groq",
    "ChatOllama": "langchain_ollama",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDER_CLASSES:
        value = getattr(import_module(_PROVIDER_CLASSES[name]), name)
    elif name == "FakeToolModel":
        value = _define_fake_tool_model()
    elif name == "ModelT":
        value = (
            _provider("AzureChatOpenAI")
            | _provider("ChatOpenAI")
            | _provider("ChatAnthropic")
            | _provider("ChatGoogleGenerativeAI")
            | _provider("ChatVertexAI")
            | _provider("ChatGroq")
            | _provider("ChatBedrock")
            | _provider("ChatOllama")
            | _provider("FakeToolModel")
        )
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # resolve once; later lookups are ordinary attributes
    return value


def _provider(name: str) -> Any:
    """A provider class by name, through the module attribute so a patch is honoured."""
    return getattr(sys.modules[__name__], name)


def _define_fake_tool_model() -> type:
    fake_list_chat_model = _provider("FakeListChatModel")

    class FakeToolModel(fake_list_chat_model):  # type: ignore[misc, valid-type]
        def __init__(self, responses: list[str]):
            super().__init__(responses=responses)

        def bind_tools(self, tools):
            return self

    FakeToolModel.__module__ = __name__
    FakeToolModel.__qualname__ = "FakeToolModel"
    return FakeToolModel


# =============================================================================
# 🎯 LLM CONFIGURATION - THE ONLY PLACE TO MODIFY MODEL SETTINGS
# =============================================================================

class LLMConfig:
    """
    🛠️ MODIFY THESE 4 VALUES TO CONTROL YOUR ENTIRE AI SYSTEM:
    """

    DEFAULT_MODEL: AllModelEnum = OpenAIModelName.GPT_4O  # Which model to use
    DEFAULT_TEMPERATURE: float = 0.0                      # 0.0=deterministic, 1.0=creative
    DEFAULT_MAX_TOKENS: int = 3000                        # Max response length
    DEFAULT_TOP_P: float = 0.9                           # Sampling diversity

    @classmethod
    def get_temperature_for_model(cls, model: AllModelEnum) -> float:
        """Get the appropriate temperature for a specific model"""
        # GPT-5 series models don't support custom temperature
        if hasattr(model, 'value') and model.value.startswith('gpt-5'):
            return 1.0  # GPT-5 only supports default temperature
        return cls.DEFAULT_TEMPERATURE

    @classmethod
    def get_max_tokens_for_model(cls, model: AllModelEnum) -> int | None:
        """Get max tokens for specific models (None = no limit)"""
        # GPT-5 series don't support max_tokens parameter
        if hasattr(model, 'value') and model.value.startswith('gpt-5'):
            return None
        return cls.DEFAULT_MAX_TOKENS

# =============================================================================

_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in OpenRouterModelName}
    | {m: m.value for m in FakeModelName}
)


# Removed get_gpt5_model function - using standard get_model instead for better performance


@cache
def get_model(model_name: AllModelEnum | None = None, /) -> "ModelT":
    # Use centralized configuration if no model specified
    if model_name is None:
        model_name = LLMConfig.DEFAULT_MODEL

    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    # Get temperature from centralized config
    temperature = LLMConfig.get_temperature_for_model(model_name)
    max_tokens = LLMConfig.get_max_tokens_for_model(model_name)

    if model_name in OpenAIModelName:
        # Use centralized config for all OpenAI models
        if max_tokens:
            return _provider("ChatOpenAI")(model=api_model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
        else:
            return _provider("ChatOpenAI")(model=api_model_name, temperature=temperature, streaming=True)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return _provider("ChatOpenAI")(
            model=settings.COMPATIBLE_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        return _provider("AzureChatOpenAI")(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
    if model_name in DeepseekModelName:
        return _provider("ChatOpenAI")(
            model=api_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in AnthropicModelName:
        return _provider("ChatAnthropic")(model=api_model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
    if model_name in GoogleModelName:
        return _provider("ChatGoogleGenerativeAI")(model=api_model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
    if model_name in VertexAIModelName:
        return _provider("ChatVertexAI")(model=api_model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
    if model_name in GroqModelName:
        # Use temperature 0.0 for LlamaGuard (deterministic), otherwise use default
        guard_temp = 0.0 if model_name == GroqModelName.LLAMA_GUARD_4_12B else temperature
        return _provider("ChatGroq")(model=api_model_name, temperature=guard_temp, max_tokens=max_tokens)
    if model_name in AWSModelName:
        return _provider("ChatBedrock")(model_id=api_model_name, temperature=temperature, max_tokens=max_tokens)
    if model_name in OllamaModelName:
        chat_ollama_class = _provider("ChatOllama")
        if settings.OLLAMA_BASE_URL:
            chat_ollama = chat_ollama_class(
                model=settings.OLLAMA_MODEL, temperature=temperature, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = chat_ollama_class(model=settings.OLLAMA_MODEL, temperature=temperature)
        return chat_ollama
    if model_name in OpenRouterModelName:
        return _provider("ChatOpenAI")(
            model=api_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
        )
    if model_name in FakeModelName:
        return _provider("FakeToolModel")(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")


# The provider class `get_model` builds for each model family, in the same order as
# its branches (membership is by value, so order decides overlapping values). Used
# only to load one provider's SDK ahead of time; a test pins it to `get_model`.
_FAMILY_PROVIDER_CLASS: tuple[tuple[Any, str], ...] = (
    (OpenAIModelName, "ChatOpenAI"),
    (OpenAICompatibleName, "ChatOpenAI"),
    (AzureOpenAIModelName, "AzureChatOpenAI"),
    (DeepseekModelName, "ChatOpenAI"),
    (AnthropicModelName, "ChatAnthropic"),
    (GoogleModelName, "ChatGoogleGenerativeAI"),
    (VertexAIModelName, "ChatVertexAI"),
    (GroqModelName, "ChatGroq"),
    (AWSModelName, "ChatBedrock"),
    (OllamaModelName, "ChatOllama"),
    (OpenRouterModelName, "ChatOpenAI"),
    (FakeModelName, "FakeToolModel"),
)


def model_provider_class_name(model_name: AllModelEnum | None = None) -> str:
    """The provider class `get_model(model_name)` builds, without building it."""
    if model_name is None:
        model_name = LLMConfig.DEFAULT_MODEL
    for family, class_name in _FAMILY_PROVIDER_CLASS:
        if model_name in family:
            return class_name
    raise ValueError(f"Unsupported model: {model_name}")


def preload_model_provider(model_name: AllModelEnum | None = None) -> str:
    """Import the SDK `get_model(model_name)` needs — the default model's if None.

    Loads the class only: no model is constructed and nothing is called.
    """
    class_name = model_provider_class_name(model_name)
    _provider(class_name)
    return class_name


# =============================================================================
# Embeddings — Resume RAG
# =============================================================================

# `text-embedding-3-small` at its native 1536 dimensions. Changing either changes
# every stored vector and every cache key built from them (see
# agents/xbuddy/resume/embeddings.py), so both are named once, here.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


@cache
def get_embeddings() -> "OpenAIEmbeddings":
    """The embedding model, shared like `get_model()`.

    Callers import this inside the function that uses it (`from core.llm import
    get_embeddings`), the same way the nodes import `get_model`, so the test suite's
    autouse guard can replace it and no test can reach the API by accident.

    The key is passed from settings when it is set. `OpenAIEmbeddings` reads the
    environment on its own otherwise, and refuses to construct without a key — which
    is the right failure for a caller that forgot to configure one.
    """
    kwargs: dict = {"model": EMBEDDING_MODEL, "dimensions": EMBEDDING_DIMENSIONS}
    if settings.OPENAI_API_KEY is not None:
        kwargs["api_key"] = settings.OPENAI_API_KEY
    return _provider("OpenAIEmbeddings")(**kwargs)
