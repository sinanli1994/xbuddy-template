from functools import cache
try:
    from typing import TypeAlias
except ImportError:
    # Python 3.9 compatibility
    TypeAlias = str

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings

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


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self


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


# Removed get_gpt5_model function - using standard get_model instead for better performance


@cache
def get_model(model_name: AllModelEnum | None = None, /) -> ModelT:
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
            return ChatOpenAI(model=api_model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
        else:
            return ChatOpenAI(model=api_model_name, temperature=temperature, streaming=True)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
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

        return AzureChatOpenAI(
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
        return ChatOpenAI(
            model=api_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
    if model_name in VertexAIModelName:
        return ChatVertexAI(model=api_model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
    if model_name in GroqModelName:
        # Use temperature 0.0 for LlamaGuard (deterministic), otherwise use default
        guard_temp = 0.0 if model_name == GroqModelName.LLAMA_GUARD_4_12B else temperature
        return ChatGroq(model=api_model_name, temperature=guard_temp, max_tokens=max_tokens)
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=temperature, max_tokens=max_tokens)
    if model_name in OllamaModelName:
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL, temperature=temperature, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=temperature)
        return chat_ollama
    if model_name in OpenRouterModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
        )
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")


# =============================================================================
# Embeddings — Resume RAG
# =============================================================================

# `text-embedding-3-small` at its native 1536 dimensions. Changing either changes
# every stored vector and every cache key built from them (see
# agents/xbuddy/resume/embeddings.py), so both are named once, here.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


@cache
def get_embeddings() -> OpenAIEmbeddings:
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
    return OpenAIEmbeddings(**kwargs)
