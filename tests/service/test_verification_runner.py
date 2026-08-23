"""The verification script must start on an event loop psycopg will accept.

Windows' default `ProactorEventLoop` makes psycopg raise `InterfaceError` *before*
it opens a connection, so a runner that forgets the policy fails in a way that looks
like a database problem and is not one.

Everything here is offline — no database, no credentials. The probe runs in a
subprocess on purpose: importing the script executes `load_dotenv`, which would push
the developer's real `DATABASE_TYPE` and `POSTGRES_URI` into `os.environ` and change
what unrelated `Settings` tests see.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_postgres_persistence.py"

PROBE = textwrap.dedent(
    """
    import asyncio, importlib.util, json, sys

    spec = importlib.util.spec_from_file_location("vpp", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def running_loop_name():
        return type(asyncio.get_running_loop()).__name__

    async def bare_loop_name():
        return type(asyncio.get_running_loop()).__name__

    runner_loop = module.run_with_compatible_loop(running_loop_name())

    print(json.dumps({
        "platform": sys.platform,
        "runner_loop": runner_loop,
        "is_proactor": runner_loop == "ProactorEventLoop",
        "win32_policy": type(module.compatible_loop_policy("win32")).__name__,
        "posix_policy": module.compatible_loop_policy("linux"),
    }))
    """
)


@pytest.fixture(scope="module")
def probe() -> dict:
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_runner_starts_on_a_loop_psycopg_accepts(probe):
    """Mirrors psycopg's own check: it rejects `ProactorEventLoop` by that exact type."""
    assert not probe["is_proactor"], f"runner produced {probe['runner_loop']}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific loop selection")
def test_windows_gets_a_selector_loop(probe):
    assert "Selector" in probe["runner_loop"], probe["runner_loop"]


@pytest.mark.skipif(sys.platform != "win32", reason="the failure only exists on Windows")
def test_without_the_fix_the_default_loop_would_be_rejected():
    """The teeth. If bare `asyncio.run` already gave a compatible loop here, the
    runner's policy would be decoration and this whole file would prove nothing."""
    default_loop = textwrap.dedent(
        """
        import asyncio
        print(type(asyncio.new_event_loop()).__name__)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", default_loop],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ProactorEventLoop"


def test_posix_behaviour_is_left_alone(probe):
    """No policy is imposed off Windows — uvloop and the default loop stay available."""
    assert probe["posix_policy"] is None


def test_the_windows_policy_is_the_selector_one(probe):
    assert probe["win32_policy"] == "WindowsSelectorEventLoopPolicy"


def test_run_service_sets_the_same_policy():
    """The real service entrypoint already handles this. Pinned so a future tidy-up
    does not quietly delete it and reintroduce the bug in the deployed runner."""
    source = (ROOT / "src" / "run_service.py").read_text(encoding="utf-8")
    assert 'sys.platform == "win32"' in source
    assert "WindowsSelectorEventLoopPolicy" in source


def test_the_script_is_still_not_collected_by_pytest():
    """It opens real connections; it must stay opt-in."""
    assert SCRIPT.name != "test_verify_postgres_persistence.py"
    assert not SCRIPT.is_relative_to(ROOT / "tests")
