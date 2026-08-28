"""Reading and renewing the session the browser already holds.

Nothing here touches the real Chrome profile, the Keychain, or the network: the
cookie store is a SQLite file built in the test, and the SSO chain is faked. That
matters more than usual for this module — its whole job is to reach outside the
process, and a test that actually did so would be untestable in CI and would
depend on whoever ran it being logged into Polar.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from polarswim import auth, browser


def _token(exp_offset_s: int) -> str:
    """A FLOW_SESSION-shaped JWT expiring `exp_offset_s` from now."""
    payload = {"sub": "1", "exp": int(time.time()) + exp_offset_s}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"header.{body}.signature"


def _cookies(exp_offset_s: int | None = 3600):
    out = [("auth.polar.com", "remember-me", "durable", "/"),
           ("auth.polar.com", "session_id", "sid-1", "/")]
    if exp_offset_s is not None:
        out.append((".flow.polar.com", "FLOW_SESSION", _token(exp_offset_s), "/"))
    return out


class TestCookieHeader:
    def test_only_flow_cookies_reach_the_header(self):
        """auth.polar.com cookies authenticate the renewal, not the API call."""
        header = browser._header(browser._jar(_cookies()))
        assert "FLOW_SESSION=" in header
        assert "remember-me" not in header

    def test_a_refreshed_cookie_shadows_the_stale_one(self):
        jar = browser._jar([(".flow.polar.com", "FLOW_SESSION", "new", "/"),
                            ("flow.polar.com", "FLOW_SESSION", "old", "/")])
        assert browser._header(jar).count("FLOW_SESSION") == 1


class TestTokenLifetime:
    def test_a_token_with_time_left_is_usable(self):
        assert browser._usable(browser._header(browser._jar(_cookies(3600))))

    def test_an_expired_token_is_not(self):
        assert not browser._usable(browser._header(browser._jar(_cookies(-60))))

    def test_a_token_about_to_die_is_not_usable(self):
        """A backfill walks the calendar in windows and outlives a two-minute
        remainder, so 'not yet expired' is the wrong test."""
        assert not browser._usable(
            browser._header(browser._jar(_cookies(browser.MIN_REMAINING_S - 30))))

    def test_no_token_at_all_is_not_usable(self):
        assert not browser._usable(browser._header(browser._jar(_cookies(None))))


class TestDecryption:
    def test_an_unrecognised_scheme_is_skipped_not_guessed(self):
        assert browser._decrypt(b"v99garbage", b"k" * 16) is None

    def test_plaintext_values_are_skipped(self):
        """Only Chrome's v10/v11 blobs are ours to interpret."""
        assert browser._decrypt(b"plain-value", b"k" * 16) is None


class TestHostRestriction:
    def test_only_polar_cookies_are_read(self, tmp_path, monkeypatch):
        """The Keychain key opens every cookie Chrome holds. A swimming tool has
        no business decrypting a bank session, so the filter is a hard boundary."""
        import sqlite3

        db = tmp_path / "Cookies"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, "
                    "encrypted_value BLOB, path TEXT)")
        con.executemany("INSERT INTO cookies VALUES (?,?,?,?)", [
            ("flow.polar.com", "FLOW_SESSION", b"v10enc", "/"),
            ("bank.example.com", "SESSION", b"v10enc", "/"),
            (".polar.com", "_ga", b"v10enc", "/"),
        ])
        con.commit()
        con.close()

        seen = []

        def fake_decrypt(blob, key):
            return "value"

        monkeypatch.setattr(browser, "_keychain_key", lambda: b"k" * 16)
        monkeypatch.setattr(browser, "cookie_db_path", lambda: db)
        monkeypatch.setattr(browser, "_decrypt", fake_decrypt)

        hosts = {host for host, _, _, _ in browser.read_cookies()}
        assert hosts == {"flow.polar.com", ".polar.com"}
        assert "bank.example.com" not in hosts


class TestCredentialFallback:
    """Two sources, each given two chances: use its token, or mint from it."""

    def test_a_live_token_is_used_without_minting(self, monkeypatch):
        minted = []
        monkeypatch.setattr(browser, "_load_store", lambda: _cookies(3600))
        monkeypatch.setattr(browser, "_mint", lambda c: minted.append(c))
        assert "FLOW_SESSION=" in browser.credential()
        assert minted == []

    def test_a_spent_token_is_renewed(self, monkeypatch):
        monkeypatch.setattr(browser, "_load_store", lambda: _cookies(-60))
        monkeypatch.setattr(browser, "_save_store", lambda c: None)
        monkeypatch.setattr(browser, "_mint",
                            lambda c: ("FLOW_SESSION=fresh", _cookies(3600)))
        assert browser.credential() == "FLOW_SESSION=fresh"

    def test_chrome_recovers_when_our_saved_session_has_gone_stale(self, monkeypatch):
        """The browser mints too, and whoever minted last holds the live session.
        A stale store must fall through to Chrome rather than giving up."""
        monkeypatch.setattr(browser, "_load_store", lambda: _cookies(-60))
        monkeypatch.setattr(browser, "read_cookies", lambda: _cookies(-60))
        monkeypatch.setattr(browser, "_save_store", lambda c: None)

        calls = []

        def mint(cookies):
            calls.append(cookies)
            if len(calls) == 1:
                raise browser.BrowserAuthError("stale")
            return "FLOW_SESSION=from-chrome", _cookies(3600)

        monkeypatch.setattr(browser, "_mint", mint)
        assert browser.credential() == "FLOW_SESSION=from-chrome"
        assert len(calls) == 2

    def test_the_error_says_what_to_do_when_both_sources_fail(self, monkeypatch):
        monkeypatch.setattr(browser, "_load_store", lambda: None)
        monkeypatch.setattr(browser, "read_cookies",
                            lambda: (_ for _ in ()).throw(
                                browser.BrowserAuthError("no chrome")))
        with pytest.raises(browser.BrowserAuthError, match="flow.polar.com in Chrome"):
            browser.credential()


class TestSourceSelection:
    """A browser session is a preference, never a requirement."""

    def test_auto_falls_back_to_the_pasted_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv(auth.ENV_VAR, raising=False)
        monkeypatch.setattr(browser, "credential",
                            lambda: (_ for _ in ()).throw(
                                browser.BrowserAuthError("declined")))
        path = tmp_path / "cookie.txt"
        path.write_text("FLOW_SESSION=pasted; other=1")
        assert "FLOW_SESSION=pasted" in auth.load_cookie(path, source="auto")

    def test_browser_only_reports_the_failure_instead_of_papering_over_it(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(browser, "credential",
                            lambda: (_ for _ in ()).throw(
                                browser.BrowserAuthError("declined")))
        path = tmp_path / "cookie.txt"
        path.write_text("FLOW_SESSION=pasted")
        with pytest.raises(auth.AuthError, match="declined"):
            auth.load_cookie(path, source="browser")

    def test_file_only_never_touches_the_browser(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.delenv(auth.ENV_VAR, raising=False)
        monkeypatch.setattr(browser, "credential", lambda: called.append(1))
        path = tmp_path / "cookie.txt"
        path.write_text("FLOW_SESSION=pasted")
        auth.load_cookie(path, source="file")
        assert called == []

    def test_an_unknown_source_is_rejected(self):
        with pytest.raises(auth.AuthError, match="unknown credential source"):
            auth.load_cookie(source="telepathy")

    def test_the_resolved_source_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.delenv(auth.ENV_VAR, raising=False)
        monkeypatch.setattr(browser, "credential", lambda: "FLOW_SESSION=live")
        auth.load_cookie(source="browser")
        assert auth.last_source["name"] == "browser"

        path = tmp_path / "cookie.txt"
        path.write_text("FLOW_SESSION=pasted")
        auth.load_cookie(path, source="file")
        assert auth.last_source["name"] == "file"
