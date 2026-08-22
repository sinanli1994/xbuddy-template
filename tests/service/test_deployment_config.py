"""PR 6 Stage 5B: the deployment artifacts, checked offline.

Two things are being defended here.

The first is that no credential ever reaches a committed file. `fly.toml` and the
`Dockerfile` are in git, so a secret placed in either is a secret published to
everyone with repository access, and the mistake is invisible in review once it has
scrolled past.

The second is that the deployed posture is the intended one — production mode on,
Postgres selected with no SQLite fallback, no browser origin allowed, no volume
pretending to be persistence. Each of those is a single line that is easy to
"tidy up" later without realising what it was load-bearing for.

Nothing here builds an image or contacts Fly. These are offline structural checks,
deliberately asserting on meaning rather than formatting, so reindenting `fly.toml`
does not turn the suite red.
"""

import re
import tomllib
from pathlib import Path

import pytest
from pydantic import SecretStr

from core.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
FLY_TOML = ROOT / "fly.toml"
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"

BASE = {"OPENAI_API_KEY": SecretStr("sk-test"), "_env_file": None}
PRODUCTION = {"MODE": "production", "AUTH_SECRET": SecretStr("token")}

# Anything credential-bearing. `SUPABASE_URL` is deliberately absent: it is the public
# project URL, not a credential, and the frontend already exposes it.
SECRET_ENV_NAMES = [
    "AUTH_SECRET",
    "OPENAI_API_KEY",
    "SUPABASE_SECRET_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_ANON_KEY",
    "POSTGRES_URI",
    "POSTGRES_PASSWORD",
    "LANGSMITH_API_KEY",
    "LANGFUSE_SECRET_KEY",
    "ANTHROPIC_API_KEY",
]


@pytest.fixture(scope="module")
def fly() -> dict:
    return tomllib.loads(FLY_TOML.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fly_env(fly) -> dict:
    return fly.get("env", {})


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# No secrets in committed deployment files
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SECRET_ENV_NAMES)
def test_no_secret_is_declared_in_fly_env(fly_env, name):
    """`fly.toml` is committed. A secret here is a published secret."""
    assert name not in fly_env, f"{name} must be a Fly secret, not fly.toml [env]"


@pytest.mark.parametrize("name", SECRET_ENV_NAMES)
def test_no_secret_is_baked_into_the_image(dockerfile, name):
    """An ENV or ARG in a Dockerfile is readable by anyone who can pull the image."""
    assert not re.search(rf"^\s*(ENV|ARG)\s+{name}\b", dockerfile, re.MULTILINE)


def test_postgres_uri_is_a_secret_not_fly_env(fly_env):
    """It embeds the database password, which is a separate credential from
    SUPABASE_SECRET_KEY — different access layer, different failure mode."""
    assert "POSTGRES_URI" not in fly_env


def test_auth_secret_is_a_secret_not_fly_env(fly_env):
    assert "AUTH_SECRET" not in fly_env


def test_no_credential_shaped_literal_appears_in_either_file(dockerfile):
    """Catches a pasted value even under a name this test does not know about."""
    haystack = dockerfile + FLY_TOML.read_text(encoding="utf-8")
    for pattern in (
        r"sk-[A-Za-z0-9]{20,}",  # OpenAI
        r"sb_secret_[A-Za-z0-9_-]{10,}",  # Supabase secret key
        r"lsv2_[A-Za-z0-9_-]{10,}",  # LangSmith
        r"eyJ[A-Za-z0-9_-]{20,}\.",  # a JWT, e.g. a legacy service-role key
        r"postgres(?:ql)?://[^\s:]+:[^\s@]+@",  # a URI with an inline password
    ):
        assert not re.search(pattern, haystack), f"credential-shaped literal: {pattern}"


def test_the_image_never_copies_an_env_file(dockerfile):
    """Only pyproject/uv.lock, src/ and scripts/ are copied. A blanket `COPY . .`
    would pull in .env from the build context regardless of .dockerignore ordering."""
    copies = re.findall(r"^\s*COPY\s+(?!--from)(.+)$", dockerfile, re.MULTILINE)
    assert copies, "expected at least one COPY"
    for line in copies:
        sources = line.split()[:-1]
        for source in sources:
            assert source not in {".", "./"}, f"blanket copy of the build context: {line}"
            assert ".env" not in source, line


def _dockerignore_patterns() -> set[str]:
    """Active patterns only — a commented-out line excludes nothing."""
    return {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_dockerignore_excludes_env_and_local_state():
    """Defence in depth behind the explicit COPYs above."""
    patterns = _dockerignore_patterns()
    for required in (".env", ".env.*", ".git", ".venv", "*.db"):
        assert required in patterns, f"missing from .dockerignore: {required}"
    assert any(p.endswith("__pycache__/") for p in patterns)
    assert any("pytest_cache" in p for p in patterns)


def test_the_deployment_probes_are_still_reachable_inside_the_image(dockerfile):
    """scripts/ is excluded wholesale, so the two diagnostics are re-included by
    negation. If that breaks, `fly ssh console` cannot run them."""
    assert re.search(r"^\s*COPY\s+scripts/", dockerfile, re.MULTILINE)
    # Substring matching would be satisfied by a commented-out `# !scripts/...`,
    # which excludes the probes while looking like it includes them. Parse instead.
    patterns = _dockerignore_patterns()
    assert "!scripts/fly_connectivity_probe.py" in patterns
    assert "!scripts/fly_durability_probe.py" in patterns
    assert "scripts/*" in patterns, "negation needs a glob, not a bare directory exclusion"


# --------------------------------------------------------------------------
# The deployed posture
# --------------------------------------------------------------------------


def test_production_mode_is_declared(fly_env):
    assert fly_env.get("MODE") == "production"


def test_the_declared_mode_actually_reads_as_production(fly_env):
    """Asserting the string alone would pass for a typo like `prod uction`."""
    assert Settings(MODE=fly_env["MODE"], AUTH_SECRET=SecretStr("t"), **BASE).is_production()


def test_postgres_is_selected(fly_env):
    assert fly_env.get("DATABASE_TYPE") == "postgres"


def test_the_declared_database_type_is_a_real_one(fly_env):
    from core.settings import DatabaseType

    assert Settings(DATABASE_TYPE=fly_env["DATABASE_TYPE"], **BASE).DATABASE_TYPE is (
        DatabaseType.POSTGRES
    )


def test_production_allows_no_browser_origin():
    """The frontend proxies server-side, so nothing needs a CORS allowance."""
    assert Settings(**PRODUCTION, **BASE).cors_allow_origins == []


def test_an_empty_cors_value_cannot_reopen_the_default():
    """`env_ignore_empty=True` makes an empty env value read as unset, so this must
    hold via the production default rather than via the fly.toml line."""
    assert Settings(**PRODUCTION, CORS_ALLOW_ORIGINS="", **BASE).cors_allow_origins == []


def test_production_cors_is_never_a_wildcard():
    assert "*" not in Settings(**PRODUCTION, **BASE).cors_allow_origins


def test_a_real_origin_still_takes_effect():
    """Empty-by-default must not mean unconfigurable."""
    settings = Settings(**PRODUCTION, CORS_ALLOW_ORIGINS="https://app.example.com", **BASE)
    assert settings.cors_allow_origins == ["https://app.example.com"]


def test_local_development_is_unaffected():
    """No MODE set: the Next.js dev server still works without configuration."""
    assert Settings(**BASE).cors_allow_origins == ["http://localhost:3000"]


def test_local_development_still_defaults_to_sqlite():
    """The deployment config must not have changed what a developer gets locally."""
    assert Settings(**BASE).DATABASE_TYPE.value == "sqlite"
    assert not Settings(**BASE).is_production()


# --------------------------------------------------------------------------
# Fly service shape
# --------------------------------------------------------------------------


def test_the_health_check_targets_health(fly):
    checks = fly["http_service"]["checks"]
    assert [c["path"] for c in checks] == ["/health"]


def test_health_is_unauthenticated():
    """The health check sends no bearer token, so a protected /health would make Fly
    kill a machine that is in fact perfectly healthy."""
    source = (ROOT / "src" / "service" / "service.py").read_text(encoding="utf-8")
    assert '@app.get("/health")' in source, "/health must be on `app`, not the guarded router"
    assert '@router.get("/health")' not in source


def test_the_internal_port_matches_the_application(fly, fly_env):
    assert fly["http_service"]["internal_port"] == int(fly_env["PORT"])


def test_the_application_defaults_to_that_port_and_binds_all_interfaces():
    settings = Settings(**BASE)
    assert settings.PORT == 8080
    assert settings.HOST == "0.0.0.0"


def test_no_volume_is_configured(fly):
    """Persistence is in Postgres. A volume would be machine-local state that a
    redeploy loses without saying so."""
    assert "mounts" not in fly


def test_it_does_not_scale_beyond_one_machine(fly):
    """slowapi holds rate-limit counters in process memory, so N machines would mean
    N independent limits."""
    assert fly["http_service"]["min_machines_running"] == 1


def test_machines_do_not_auto_stop(fly):
    """Suspending mid-stream would drop in-flight SSE connections."""
    assert fly["http_service"]["auto_stop_machines"] == "off"


def test_https_is_forced(fly):
    assert fly["http_service"]["force_https"] is True


# --------------------------------------------------------------------------
# Startup path
# --------------------------------------------------------------------------


def test_the_container_starts_through_the_existing_entrypoint(dockerfile):
    """Not a second startup path. run_service.py already reads PORT, binds HOST, and
    carries the Windows event-loop policy."""
    assert re.search(r'CMD\s+\["python",\s*"src/run_service\.py"\]', dockerfile)


def test_the_entrypoint_honours_flys_port_and_binds_all_interfaces():
    source = (ROOT / "src" / "run_service.py").read_text(encoding="utf-8")
    assert 'os.environ.get("PORT"' in source
    assert "host=settings.HOST" in source


def test_the_windows_loop_policy_is_preserved_and_not_made_linux_specific():
    """It must stay guarded by sys.platform so the Linux container is unaffected."""
    source = (ROOT / "src" / "run_service.py").read_text(encoding="utf-8")
    assert 'sys.platform == "win32"' in source
    assert "WindowsSelectorEventLoopPolicy" in source


def test_dependencies_install_reproducibly(dockerfile):
    """--frozen fails if uv.lock and pyproject.toml disagree; without it the build
    could silently resolve something the lockfile never pinned."""
    assert re.search(r"uv sync[^\n]*--frozen", dockerfile)
    assert re.search(r"uv sync[^\n]*--no-dev", dockerfile)


def test_the_uv_image_is_pinned(dockerfile):
    """`:latest` would make the image non-reproducible."""
    match = re.search(r"COPY --from=ghcr\.io/astral-sh/uv:(\S+)", dockerfile)
    assert match, "expected a uv stage"
    assert match.group(1) != "latest"


def test_the_base_image_is_a_supported_python_matching_requires_python(dockerfile):
    match = re.search(r"^FROM python:(\d+)\.(\d+)-slim", dockerfile, re.MULTILINE)
    assert match, "expected a slim python base"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (3, 11), "below requires-python >=3.11"


def test_the_container_runs_as_a_non_root_user(dockerfile):
    assert re.search(r"^\s*USER\s+(?!root\b)\S+", dockerfile, re.MULTILINE)
    user_index = dockerfile.index("\nUSER ")
    assert user_index > dockerfile.index("COPY src/"), "USER must come after the copies"


# --------------------------------------------------------------------------
# Production startup safety
# --------------------------------------------------------------------------


def test_production_refuses_to_start_without_auth_secret():
    """verify_bearer returns early when AUTH_SECRET is unset — fine locally, wide open
    in a deployment. The process must refuse to boot rather than serve unprotected."""
    with pytest.raises(ValueError, match="AUTH_SECRET"):
        Settings(MODE="production", **BASE)


def test_production_starts_when_auth_secret_is_present():
    assert Settings(**PRODUCTION, **BASE).is_production()


def test_local_runs_still_need_no_auth_secret():
    Settings(**BASE)  # must not raise
