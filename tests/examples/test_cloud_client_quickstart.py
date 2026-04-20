"""Run examples/cloud_client.py against a live AgentServer."""

from __future__ import annotations

import pytest

from examples import cloud_client

pytestmark = pytest.mark.integration


def test_quickstart_returns_zero_against_live_server(echo_server_url):
    assert cloud_client.main(echo_server_url) == 0
