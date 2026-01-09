"""
Pytest configuration for AgentBusters tests.
"""

import pytest


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://localhost:9109",
        help="URL of the Green Agent to test"
    )
    parser.addoption(
        "--purple-url",
        action="store",
        default="http://localhost:9110",
        help="URL of the Purple Agent for integration tests"
    )


@pytest.fixture
def agent(request):
    """Get the Green Agent URL from command line."""
    return request.config.getoption("--agent-url")


@pytest.fixture
def purple_agent(request):
    """Get the Purple Agent URL from command line."""
    return request.config.getoption("--purple-url")

