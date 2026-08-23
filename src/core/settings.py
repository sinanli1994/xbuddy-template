from enum import StrEnum
from json import loads
from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    Provider,
    VertexAIModelName,
)


class DatabaseType(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGO = "mongo"


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


DEV_BROWSER_ORIGIN = "http://localhost:3000"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8080, description="Port to run the server on")

    AUTH_SECRET: SecretStr | None = None

    # PR 6. Comma-separated browser origins allowed to call this API directly.
    # Replaces the previous `allow_origins=["*"]`. The default covers local Next.js
    # development only; a deployed frontend origin is supplied through the
    # environment so no deployment URL is baked into the repository.
    # Comma-separated browser origins. Left unset it means "the Next.js dev server"
    # locally and "no browser origin at all" in production — see `cors_allow_origins`.
    # Note `env_ignore_empty=True` above: an empty env value reads as *unset*, so a
    # deployment cannot clear this by setting it to "". That is why the production
    # default has to be empty rather than something that must be overridden.
    CORS_ALLOW_ORIGINS: str = ""

    # PR 6. Rate limit for the two expensive LLM entrypoints (/invoke, /stream).
    # Conservative demo default: a public URL behind a shared token exposes real
    # model spend, and this is the cheapest control that bounds it.
    RATE_LIMIT_EXPENSIVE: str = "10/minute"

    OPENAI_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    USE_AWS_BEDROCK: bool = False
    OLLAMA_MODEL: str | None = None
    OLLAMA_BASE_URL: str | None = None
    USE_FAKE_MODEL: bool = False
    OPENROUTER_API_KEY: str | None = None

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    # Set openai compatible api, mainly used for proof of concept
    COMPATIBLE_MODEL: str | None = None
    COMPATIBLE_API_KEY: SecretStr | None = None
    COMPATIBLE_BASE_URL: str | None = None

    OPENWEATHERMAP_API_KEY: SecretStr | None = None


    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_TRACING: bool = False
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None

    # Database Configuration
    DATABASE_TYPE: DatabaseType = (
        DatabaseType.SQLITE
    )  # Options: DatabaseType.SQLITE or DatabaseType.POSTGRES
    SQLITE_DB_PATH: str = "/tmp/checkpoints.db"

    # PostgreSQL Configuration
    #
    # PR 6. `POSTGRES_URI` is the deployment-facing form: one libpq URI, which is
    # exactly what Supabase hands you and what psycopg's pool already takes as its
    # `conninfo`. It wins when set. The five discrete fields below remain the local
    # and backwards-compatible path — nothing that already works stops working.
    #
    # Held as a SecretStr because the URI embeds the password. Include
    # `?sslmode=require`; this code never rewrites the URI it is given.
    POSTGRES_URI: SecretStr | None = None
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None

    # MongoDB Configuration
    MONGO_HOST: str | None = None
    MONGO_PORT: int | None = None
    MONGO_DB: str | None = None
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_AUTH_SOURCE: str | None = None

    # Supabase Configuration
    #
    # Two generations of API keys are supported. The newer names take precedence
    # (see integrations/supabase/supabase_client._resolve_backend_key); the legacy
    # JWT names are kept so existing deployments keep working.
    #
    #   SUPABASE_SECRET_KEY       (sb_secret_...)      replaces SERVICE_ROLE_KEY
    #   SUPABASE_PUBLISHABLE_KEY  (sb_publishable_...) replaces ANON_KEY
    #
    # These must be declared here to have any effect: model_config sets
    # extra="ignore", so an undeclared name in .env is silently dropped.
    SUPABASE_URL: str | None = None
    SUPABASE_SECRET_KEY: SecretStr | None = None
    SUPABASE_PUBLISHABLE_KEY: SecretStr | None = None
    SUPABASE_ANON_KEY: SecretStr | None = None
    SUPABASE_SERVICE_ROLE_KEY: SecretStr | None = None
    # DEAD as of PR 6 — declared, but read by nothing. Setting it does NOT configure
    # Postgres; the service would still fail `validate_postgres_config()` and refuse to
    # start. Use `POSTGRES_URI` (or the five discrete POSTGRES_* fields) instead.
    # Kept only so an existing .env carrying it does not become an unknown key.
    SUPABASE_DB_URL: str | None = None
    USE_SUPABASE_REALTIME: bool = False

    # Note: LLM configuration moved to src/core/llm_config.py

    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_MAP: dict[str, str] = Field(
        default_factory=dict, description="Map of model names to Azure deployment IDs"
    )

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_BASE_URL and self.COMPATIBLE_MODEL,
            Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.VERTEXAI: self.GOOGLE_APPLICATION_CREDENTIALS,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.AWS: self.USE_AWS_BEDROCK,
            Provider.OLLAMA: self.OLLAMA_MODEL,
            Provider.FAKE: self.USE_FAKE_MODEL,
            Provider.AZURE_OPENAI: self.AZURE_OPENAI_API_KEY,
            Provider.OPENROUTER: self.OPENROUTER_API_KEY,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_CHAT
                    self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.HAIKU_3
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case Provider.VERTEXAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = VertexAIModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(VertexAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.AWS:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                    self.AVAILABLE_MODELS.update(set(AWSModelName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                case Provider.OPENROUTER:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenRouterModelName.GEMINI_25_FLASH
                    self.AVAILABLE_MODELS.update(set(OpenRouterModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case Provider.AZURE_OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AzureOpenAIModelName.AZURE_GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(AzureOpenAIModelName))
                    # Validate Azure OpenAI settings if Azure provider is available
                    if not self.AZURE_OPENAI_API_KEY:
                        raise ValueError("AZURE_OPENAI_API_KEY must be set")
                    if not self.AZURE_OPENAI_ENDPOINT:
                        raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
                    if not self.AZURE_OPENAI_DEPLOYMENT_MAP:
                        raise ValueError("AZURE_OPENAI_DEPLOYMENT_MAP must be set")

                    # Parse deployment map if it's a string
                    if isinstance(self.AZURE_OPENAI_DEPLOYMENT_MAP, str):
                        try:
                            self.AZURE_OPENAI_DEPLOYMENT_MAP = loads(
                                self.AZURE_OPENAI_DEPLOYMENT_MAP
                            )
                        except Exception as e:
                            raise ValueError(f"Invalid AZURE_OPENAI_DEPLOYMENT_MAP JSON: {e}")

                    # Validate required deployments exist
                    required_models = {"gpt-4o", "gpt-4o-mini"}
                    missing_models = required_models - set(self.AZURE_OPENAI_DEPLOYMENT_MAP.keys())
                    if missing_models:
                        raise ValueError(f"Missing required Azure deployments: {missing_models}")
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.MODE == "dev"

    def is_production(self) -> bool:
        """Whether this process is running as a deployed service.

        Production is an **explicit opt-in**, not the absence of dev. `MODE` is
        unset by default, so treating "not dev" as production would make every
        local run fail the AUTH_SECRET check below — the opposite of the intended
        developer convenience. Both spellings are accepted because
        `tests/core/test_settings.py` already uses "prod".
        """
        return (self.MODE or "").strip().lower() in {"prod", "production"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_allow_origins(self) -> list[str]:
        """`CORS_ALLOW_ORIGINS` parsed into a list, blanks dropped.

        Never returns `["*"]` by accident. When nothing is configured the answer
        depends on the mode, and the production half is the one that matters:

        * **production** -> `[]`, denying every browser origin. The deployed
          architecture is browser -> Vercel -> server-side proxy -> this API, so no
          browser ever sends us a cross-origin request and there is nothing to allow.
        * **anything else** -> the local Next.js dev server, so working locally needs
          no ceremony.

        The mode check is load-bearing rather than cosmetic. `env_ignore_empty=True`
        means `CORS_ALLOW_ORIGINS=""` in a deployment config is read as *unset*, so
        without this the field would fall back to its literal default and a
        production API would quietly advertise `http://localhost:3000` as an allowed
        origin — with the deployment config that looks like it set otherwise sitting
        right there, inert.
        """
        configured = [
            origin.strip()
            for origin in (self.CORS_ALLOW_ORIGINS or "").split(",")
            if origin.strip()
        ]
        if configured:
            return configured
        return [] if self.is_production() else [DEV_BROWSER_ORIGIN]

    @model_validator(mode="after")
    def _require_auth_secret_in_production(self) -> "Settings":
        """A deployed service must never start with authentication disabled.

        `verify_bearer` returns early when `AUTH_SECRET` is unset, which is
        deliberate for local development — no token juggling while iterating. In a
        deployed process that same branch would leave every route open to the
        internet, and it would do so silently.

        So the check is here, at configuration time, rather than as a second auth
        mechanism: the process refuses to start instead of serving an unprotected
        API. Local runs are unaffected because production is an explicit opt-in.
        """
        if self.is_production() and self.AUTH_SECRET is None:
            raise ValueError(
                "AUTH_SECRET is required when MODE is production. "
                "Refusing to start with authentication disabled on a deployed "
                "service. Set AUTH_SECRET, or unset MODE for local development."
            )
        return self


settings = Settings()
