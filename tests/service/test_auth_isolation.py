"""Service tests must not inherit the developer's deployment secrets.

Adding a real `AUTH_SECRET` to `.env` for the Fly deployment took this suite from 15
failures to 33 without a single line of application code changing. `core/settings.py`
builds its `Settings()` singleton at import time from `find_dotenv()`, `service.service`
binds that instance, and `verify_bearer` reads `settings.AUTH_SECRET` on every request —
so eighteen tests that send no bearer token started getting 401.

`conftest.deterministic_auth` pins that state for ordinary tests. These tests prove the
pin actually works, that it is not vacuous, and — most importantly — that it does not
quietly defang the tests whose whole job is to prove authentication works.

All offline. Nothing here reads or writes `.env`.
"""

import pytest
from pydantic import SecretStr

import service.service as service_module
from core.settings import Settings


def dotenv_carries_a_secret() -> bool:
    """Whether the developer's real `.env` sets AUTH_SECRET, read without touching the
    runtime singleton."""
    return Settings().AUTH_SECRET is not None


# --------------------------------------------------------------------------
# The pin holds, and is not vacuous
# --------------------------------------------------------------------------


def test_ordinary_tests_see_authentication_disabled():
    """The object production code reads — not a copy, not os.environ."""
    assert service_module.settings.AUTH_SECRET is None


def test_the_pin_is_doing_real_work_when_dotenv_has_a_secret():
    """Guards against the fix being a no-op on the machine that needed it.

    If `.env` has no AUTH_SECRET this asserts nothing — and says so — rather than
    passing silently and implying a guarantee it did not check.
    """
    if not dotenv_carries_a_secret():
        pytest.skip("this developer's .env sets no AUTH_SECRET; nothing to isolate from")
    assert service_module.settings.AUTH_SECRET is None, (
        "the runtime settings object still carries the .env secret"
    )


def test_an_unauthenticated_request_is_not_rejected(test_client, mock_agent):
    """The behaviour the 18 failures were about: no token, no 401."""
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code != 401


def test_a_stray_bearer_token_is_also_accepted_when_auth_is_off(test_client, mock_agent):
    response = test_client.post(
        "/invoke", json={"message": "test"}, headers={"Authorization": "Bearer anything"}
    )
    assert response.status_code != 401


# --------------------------------------------------------------------------
# Opting back in: the `auth_secret` fixture
# --------------------------------------------------------------------------


def test_requesting_auth_secret_turns_authentication_on(auth_secret):
    """Requesting the fixture suppresses the autouse pin, so the test controls the
    state rather than inheriting it."""
    assert service_module.settings.AUTH_SECRET is not None
    assert service_module.settings.AUTH_SECRET.get_secret_value() == auth_secret


def test_no_token_is_401_when_the_test_enables_auth(auth_secret, test_client, mock_agent):
    assert test_client.post("/invoke", json={"message": "test"}).status_code == 401


def test_a_wrong_token_is_401_when_the_test_enables_auth(auth_secret, test_client, mock_agent):
    response = test_client.post(
        "/invoke", json={"message": "test"}, headers={"Authorization": "Bearer wrong"}
    )
    assert response.status_code == 401


def test_the_correct_token_is_accepted(auth_secret, test_client, mock_agent):
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": f"Bearer {auth_secret}"},
    )
    assert response.status_code != 401


def test_the_enabled_secret_is_the_fixtures_own_not_the_developers(auth_secret):
    """It must come from the fixture, so the boundary tests behave identically on a
    machine whose `.env` has a different secret — or none."""
    assert service_module.settings.AUTH_SECRET.get_secret_value() == auth_secret

    from_dotenv = Settings().AUTH_SECRET
    if from_dotenv is not None:
        assert auth_secret != from_dotenv.get_secret_value(), (
            "the fixture must install its own secret, not the developer's"
        )


# --------------------------------------------------------------------------
# /health stays public either way
# --------------------------------------------------------------------------


def test_health_is_public_with_auth_disabled(test_client):
    assert test_client.get("/health").status_code == 200


def test_health_is_public_with_auth_enabled(auth_secret, test_client):
    """Fly's health check sends no bearer token; a protected /health would make it kill
    a machine that is in fact healthy."""
    assert test_client.get("/health").status_code == 200


# --------------------------------------------------------------------------
# The pin must not survive the test that set it
# --------------------------------------------------------------------------


def test_the_pin_is_restored_between_tests(auth_secret):
    """monkeypatch undoes both fixtures, so no test can leak auth state into the next
    one — this runs after the tests above and still sees the fixture's value."""
    assert service_module.settings.AUTH_SECRET.get_secret_value() == auth_secret


def test_and_the_next_test_is_back_to_disabled():
    assert service_module.settings.AUTH_SECRET is None


def test_the_real_singleton_is_the_one_being_patched():
    """`service.service.settings` and `core.settings.settings` must be the same object,
    or the pin would patch something production never reads."""
    from core.settings import settings as core_singleton

    assert service_module.settings is core_singleton


def test_production_fail_fast_is_untouched_by_the_isolation():
    """The pin must not make the deployed-safety guarantee unprovable."""
    with pytest.raises(ValueError, match="AUTH_SECRET is required"):
        Settings(MODE="production", OPENAI_API_KEY=SecretStr("sk-x"), _env_file=None)
