import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def pool_swim_payload():
    return _load("details_pool_swim.json")


@pytest.fixture
def run_payload():
    return _load("details_run.json")


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch):
    """Keep the whole suite away from the real Chrome profile and the network.

    `load_cookie` defaults to asking the browser, so without this any test that
    resolves a credential would read the cookie store of whoever ran it — passing
    on the author's machine, prompting or failing on anyone else's, and quietly
    testing the wrong code path in both cases.

    Only the two primitives that actually reach outside the process are blocked,
    plus the saved session, so `credential()`'s own fallback logic stays testable.
    Tests that exercise it patch these themselves, which overrides this.
    """
    from polarswim import browser

    def no_keychain():
        raise browser.BrowserAuthError("Keychain access is disabled in tests")

    def no_network(cookies):
        raise browser.BrowserAuthError("network access is disabled in tests")

    monkeypatch.setattr(browser, "_keychain_key", no_keychain)
    monkeypatch.setattr(browser, "_mint", no_network)
    monkeypatch.setattr(browser, "_load_store", lambda: None)
