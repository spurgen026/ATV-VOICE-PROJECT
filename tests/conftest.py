"""Shared fixtures. Model-loading fixtures are session-scoped so the
router/chat LLMs load once for the whole test run, not once per test -
loading takes real seconds each time.

Slow tests (anything touching a real model) are marked @pytest.mark.slow
and skipped by default - run `pytest -m slow` or `pytest --run-slow` to
include them. Keeping the default run fast is what makes a test suite
something people actually run before every change, instead of an
occasional chore.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="also run @pytest.mark.slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow") or config.getoption("-m") == "slow":
        return
    skip_slow = pytest.mark.skip(reason="slow test - loads a real model; use --run-slow to include")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def router_llm():
    from resort_atv_voice.router import load_router_model

    return load_router_model()


@pytest.fixture(scope="session")
def router_grammar():
    from resort_atv_voice.router import load_grammar

    return load_grammar()


@pytest.fixture(scope="session")
def chat_llm():
    from resort_atv_voice.router import load_chat_model

    return load_chat_model()
