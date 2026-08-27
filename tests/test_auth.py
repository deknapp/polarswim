"""Credential handling: accepting a pasted cURL, and detecting a dead session."""

import time

import pytest

from polarswim.auth import AuthError, assert_valid, load_cookie, session_expiry

# A FLOW_SESSION-shaped JWT whose payload decodes to {"exp": 4102444800} (year 2100).
FUTURE_JWT = ("FLOW_SESSION=header."
              "eyJleHAiOiA0MTAyNDQ0ODAwfQ.sig")
PAST_JWT = ("FLOW_SESSION=header."
            "eyJleHAiOiAxMDAwMDAwMDAwfQ.sig")

CURL = (
    "curl --url 'https://flow.polar.com/api/training/analysis/1/details' \\\n"
    "  -H 'accept: */*' \\\n"
    f"  -b 'CookieConsent=x; {FUTURE_JWT}; PLAY_LANG=en' \\\n"
    "  -H 'x-requested-with: XMLHttpRequest'\n"
)


def test_extracts_cookie_from_pasted_curl(tmp_path):
    p = tmp_path / "auth.txt"
    p.write_text(CURL)
    cookie = load_cookie(p)
    assert cookie.startswith("CookieConsent=x;")
    assert "FLOW_SESSION=" in cookie
    assert "curl" not in cookie


def test_accepts_a_bare_cookie_string(tmp_path, monkeypatch):
    monkeypatch.setenv("POLAR_COOKIE", f"a=1; {FUTURE_JWT}")
    assert "FLOW_SESSION=" in load_cookie()


def test_rejects_credential_from_the_wrong_host(tmp_path):
    """The localizations host serves no cookies — a common copy-paste mistake."""
    p = tmp_path / "auth.txt"
    p.write_text("curl --url 'https://localizations.flow.polar.com/x.json' "
                 "-H 'accept: */*'")
    with pytest.raises(AuthError, match="FLOW_SESSION"):
        load_cookie(p)


def test_missing_credential_explains_how_to_get_one(tmp_path):
    with pytest.raises(AuthError, match="Copy as cURL"):
        load_cookie(tmp_path / "nope.txt")


def test_reads_expiry_from_the_jwt():
    assert session_expiry(FUTURE_JWT) == 4102444800.0


def test_expiry_is_none_when_unreadable():
    assert session_expiry("FLOW_SESSION=not-a-jwt") is None


def test_expired_session_is_caught_before_any_request():
    with pytest.raises(AuthError, match="expired"):
        assert_valid(PAST_JWT)


def test_valid_session_passes():
    assert_valid(FUTURE_JWT)      # does not raise
