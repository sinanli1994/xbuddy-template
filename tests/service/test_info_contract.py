"""`/info` must not advertise routes that do not exist.

The handlers for `/sync_section` and `/refine_section` were deleted in PR 6, but their
`EndpointInfo` entries stayed in `/info`. A live probe confirmed the mismatch: `/info`
listed both, and calling either returned 404. It also cost real diagnostic time — the
stale list was the first thing that looked like evidence of a stale deployment, and it
was not.
"""

import pytest
from fastapi.routing import APIRoute

from service import app

REMOVED_PATHS = ["/sync_section", "/refine_section", "/business_plan", "/generate_business_plan"]


@pytest.fixture(scope="module")
def registered_paths() -> set[str]:
    return {route.path for route in app.routes if isinstance(route, APIRoute)}


def test_every_advertised_endpoint_is_actually_registered(test_client, registered_paths):
    """The general rule, so a future addition cannot drift either."""
    body = test_client.get("/info").json()
    for entry in body["endpoints"]:
        assert entry["path"] in registered_paths, f"/info advertises unregistered {entry['path']}"


@pytest.mark.parametrize("path", REMOVED_PATHS)
def test_deleted_routes_are_not_advertised(path, test_client):
    """Pins the specific regression, by path prefix — the stale entries carried
    parameterised suffixes like /sync_section/{agent_id}/{section_id}."""
    body = test_client.get("/info").json()
    assert not [e for e in body["endpoints"] if e["path"].startswith(path)]


@pytest.mark.parametrize("path", REMOVED_PATHS)
def test_deleted_routes_are_not_registered(path, registered_paths):
    assert not [p for p in registered_paths if p.startswith(path)]


def test_info_still_reports_the_agent_and_default(test_client):
    """Removing stale entries must not gut the endpoint."""
    body = test_client.get("/info").json()
    assert body["default_agent"] == "xbuddy"
    assert [a["key"] for a in body["agents"]] == ["xbuddy"]


def test_the_live_endpoints_clients_use_are_registered(registered_paths):
    """These are what the frontend proxy actually calls."""
    for path in ("/invoke", "/stream", "/history", "/info", "/health"):
        assert path in registered_paths
