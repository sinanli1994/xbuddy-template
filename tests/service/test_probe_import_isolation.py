"""The deployment probes must load configuration without any LLM provider present.

A Postgres diagnostic failed on Fly with `ModuleNotFoundError: No module named
'langchain_anthropic'` — a provider it never uses — before it could say anything about
the database. The cause was import coupling: `core/__init__.py` eagerly imported
`core.llm`, which imports eight provider SDKs at module scope, and Python runs a
package's `__init__` before any submodule. So `import core.settings` pulled in every
LLM provider in the project.

That mattered beyond the probe. `langchain_anthropic` is the *first* third-party
import in the whole chain — `core/llm.py` is stdlib-only until line 8, and it does not
reach `from core.settings import settings` until line 32 — so **any** interpreter
missing site-packages fails there. A wrong-interpreter problem and a missing-dependency
problem produce a byte-identical error message, which is what made the failure hard to
read.

These tests run in subprocesses that make the providers genuinely unimportable, which
in-process monkeypatching cannot fake once a module is already in `sys.modules`.
Nothing here contacts a database or Fly.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Every provider core.llm imports at module scope.
PROVIDERS = [
    "langchain_anthropic",
    "langchain_aws",
    "langchain_google_genai",
    "langchain_google_vertexai",
    "langchain_groq",
    "langchain_ollama",
]

BLOCKER = """
import sys

class Blocked:
    \"\"\"A meta-path finder that makes the named packages genuinely unimportable,
    reproducing a deployment where they were never installed.\"\"\"

    def __init__(self, names):
        self.names = set(names)

    def find_module(self, fullname, path=None):
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ModuleNotFoundError(f"No module named {fullname!r}")
        return None

sys.meta_path.insert(0, Blocked(BLOCK_LIST))
for name in list(sys.modules):
    if name.split(".")[0] in set(BLOCK_LIST):
        del sys.modules[name]
"""


def run_isolated(body: str, block: list[str]) -> subprocess.CompletedProcess:
    script = (
        f"BLOCK_LIST = {block!r}\n"
        + BLOCKER
        + "\nimport sys\nsys.path.insert(0, r'"
        + str(ROOT / "src")
        + "')\n"
        + textwrap.dedent(body)
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_the_blocker_actually_blocks():
    """Control. If the providers were importable anyway, every test below would pass
    for the wrong reason."""
    result = run_isolated(
        """
        try:
            import langchain_anthropic
            print("IMPORTED")
        except ModuleNotFoundError:
            print("BLOCKED")
        """,
        PROVIDERS,
    )
    assert result.stdout.strip().endswith("BLOCKED"), result.stdout + result.stderr


def test_settings_import_no_longer_needs_any_llm_provider():
    """The regression itself: configuration must load on its own."""
    result = run_isolated(
        """
        from core.settings import DatabaseType, settings
        print("OK", settings.DATABASE_TYPE is not None, DatabaseType.POSTGRES.value)
        """,
        PROVIDERS,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK True postgres")


@pytest.mark.parametrize(
    "probe", ["fly_connectivity_probe.py", "fly_durability_probe.py", "verify_postgres_persistence.py"]
)
def test_each_probe_imports_without_the_providers(probe):
    """Importing the module runs everything above `if __name__ == '__main__'`, which is
    where the failing import chain lived."""
    result = run_isolated(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "probe", r"{ROOT / 'scripts' / probe}"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("IMPORTED OK")
        """,
        PROVIDERS,
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORTED OK" in result.stdout


def test_the_connectivity_probe_reaches_its_database_check_without_providers():
    """Not just importable — it must get far enough to report on the database. With
    DATABASE_TYPE unset it should reach its own guard and say so, rather than dying on
    an unrelated provider import."""
    result = run_isolated(
        """
        import os
        os.environ["DATABASE_TYPE"] = "sqlite"
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "probe", r"{path}"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raise SystemExit(module.run())
        """.replace("{path}", str(ROOT / "scripts" / "fly_connectivity_probe.py")),
        PROVIDERS,
    )
    assert "not postgres" in result.stdout, result.stdout + result.stderr


def test_core_still_exposes_get_model():
    """Laziness must not remove the public API. `from core import get_model` has to keep
    working — it just resolves on use now."""
    result = run_isolated(
        """
        from core import get_model
        print("CALLABLE", callable(get_model))
        """,
        [],  # providers available here; get_model genuinely needs them
    )
    assert result.returncode == 0, result.stderr
    assert "CALLABLE True" in result.stdout


def test_importing_settings_does_not_load_core_llm():
    """The mechanism, pinned directly."""
    result = run_isolated(
        """
        import sys
        import core.settings
        print(json.dumps({
            "core_llm": "core.llm" in sys.modules,
            "anthropic": "langchain_anthropic" in sys.modules,
        }))
        """.replace("import sys\n", "import json, sys\n"),
        [],
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == {"core_llm": False, "anthropic": False}


def test_get_model_still_pulls_in_the_provider_when_used():
    """The import is deferred, not removed — asking for a model must still work."""
    result = run_isolated(
        """
        import sys
        import core
        before = "core.llm" in sys.modules
        core.get_model
        print("BEFORE", before, "AFTER", "core.llm" in sys.modules)
        """,
        [],
    )
    assert result.returncode == 0, result.stderr
    assert "BEFORE False AFTER True" in result.stdout


def test_the_service_entrypoints_config_imports_avoid_the_provider_chain():
    """The same coupling, on the hot path. `run_service.py` imports `core` and
    `core.logging_config` at module scope purely to read HOST/PORT and set up logging;
    neither should drag in eight provider SDKs.

    Asserted behaviourally rather than by reading the source: `core/__init__.py` still
    contains the string `from core.llm import get_model` twice — once under
    `TYPE_CHECKING`, once inside `__getattr__` — and neither runs at import time. A
    text search would have flagged both.
    """
    source = (ROOT / "src" / "run_service.py").read_text(encoding="utf-8")
    assert "from core import settings" in source
    assert "from core.logging_config import" in source

    result = run_isolated(
        """
        import sys
        from core import settings
        from core.logging_config import setup_logging
        print("LOADED_CORE_LLM", "core.llm" in sys.modules)
        """,
        PROVIDERS,
    )
    assert result.returncode == 0, result.stderr
    assert "LOADED_CORE_LLM False" in result.stdout
