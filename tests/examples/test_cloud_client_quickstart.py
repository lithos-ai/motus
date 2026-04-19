"""Run examples/cloud_client/quickstart.py against a live AgentServer."""

from __future__ import annotations

import pytest

from examples.cloud_client import quickstart

pytestmark = pytest.mark.integration


def test_quickstart_returns_zero_against_live_server(echo_server_url):
    assert quickstart.main(echo_server_url) == 0
