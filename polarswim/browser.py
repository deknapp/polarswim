"""Take the session from the browser that already has one, and renew it there.

Flow issues its web client a `FLOW_SESSION` JWT that lives for exactly one hour
and carries no refresh claim, so a credential copied by hand is dead by the next
morning. That hourly expiry is not, however, how the browser stays logged in.
Behind it sits a `remember-me` cookie on `auth.polar.com` with a two-week life,
and the browser silently trades it for a new hour-long token on every page load —
which is why the site never asks for a password while a hand-copied token does.

So this module copies the credential that MINTS tokens rather than a minted one:

  1. Read the `*.polar.com` cookies out of the local Chrome profile, decrypting
     them with the key Chrome keeps in the macOS Keychain.
  2. If the session token is expired or absent, replay the redirect chain the
     browser itself performs — `/flowSso/login` to `auth.polar.com` and back
     through `/flowSso/redirect` — carrying `remember-me`. Flow mints a fresh
     token and the chain ends logged in.

Step 2 completes over plain HTTP with no JavaScript, which is what makes this
worth doing: the login *page* is a JavaScript application, but the silent
re-authentication behind it is ordinary redirects and cookies, so no headless
browser is needed and the project keeps its no-extra-dependencies property.

Deliberate limits. Only hosts under `polar.com` are ever read — the decryption
key opens every cookie Chrome holds, and a tool that swims should not be
touching a bank session. Nothing here is a fallback for the pasted cookie so much
as a replacement for needing one; every failure raises and lets the caller fall
back to the file, because a broken convenience must not become a broken sync.
"""

from __future__ import annotations

import binascii
import hashlib
import http.cookiejar
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from .auth import AuthError, session_expiry

# Chrome's macOS key derivation: a Keychain password stretched with these exact
# constants, then AES-128-CBC with an all-spaces IV. They are Chrome's, not ours.
_KEYCHAIN_SERVICE = "Chrome Safe Storage"
_SALT = b"saltysalt"
_ITERATIONS = 1003
_KEY_LENGTH = 16
_IV = b" " * 16

# Only these are ever decrypted. The key would open anything Chrome has stored.
_ALLOWED_HOST_SUFFIX = "polar.com"

_PROFILE_ROOTS = (
    "~/Library/Application Support/Google/Chrome",
    "~/Library/Application Support/Chromium",
)

SSO_ENTRY = "https://flow.polar.com/flowSso/login"
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")

# Renew rather than hand back a token about to die mid-sync. A backfill walks the
# calendar in 95-day windows and can easily outlive a two-minute remainder.
MIN_REMAINING_S = 300


class BrowserAuthError(AuthError):
    """The browser could not supply a credential.

    A subclass of `AuthError` so a caller that only wants "no credential" can
    catch one type, and one that wants to fall back to the pasted file can catch
    this specifically.
    """


# --- reading Chrome's cookie store -----------------------------------------
def _keychain_key() -> bytes:
    """The AES key Chrome derives from its Keychain password.

    The first call raises a macOS authorisation dialog. Approving it once is the
    entire manual cost of this path, replacing a copy-paste per hour.
    """
    if sys.platform != "darwin":
        raise BrowserAuthError(
            "Reading the browser session is implemented for macOS only; "
            "save a copied cURL to the cookie file instead.")
    proc = subprocess.run(
        ["security", "find-generic-password", "-w", "-s", _KEYCHAIN_SERVICE],
        capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise BrowserAuthError(
            "Could not read Chrome's key from the Keychain — the prompt may have "
            "been declined. Approve it, or fall back to the pasted cookie file.")
    return hashlib.pbkdf2_hmac("sha1", proc.stdout.strip().encode(),
                               _SALT, _ITERATIONS, _KEY_LENGTH)


def cookie_db_path() -> Path:
    """The most recently written Chrome cookie store across all profiles.

    A person with several Chrome profiles is logged into Flow in one of them, and
    the one they used last is overwhelmingly the one that has the session.
    """
    found = []
    for root in _PROFILE_ROOTS:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        found += [p for p in base.glob("*/Cookies") if p.is_file()]
        found += [p for p in base.glob("*/Network/Cookies") if p.is_file()]
    if not found:
        raise BrowserAuthError(
            "No Chrome cookie store found. Is Chrome installed and signed in to "
            "flow.polar.com?")
    return max(found, key=lambda p: p.stat().st_mtime)


def _decrypt(blob: bytes, key: bytes) -> str | None:
    """One cookie value, or None if it is not in a format we understand."""
    if blob[:3] not in (b"v10", b"v11"):
        return None                          # unencrypted or a scheme we don't know
    proc = subprocess.run(
        ["openssl", "enc", "-aes-128-cbc", "-d", "-nopad",
         "-K", binascii.hexlify(key).decode(), "-iv", binascii.hexlify(_IV).decode()],
        input=blob[3:], capture_output=True)
    if proc.returncode != 0:
        return None
    plain = proc.stdout
    if plain and plain[-1] <= 16:
        plain = plain[:-plain[-1]]           # PKCS#7, stripped by hand under -nopad
    # Chrome 127 and later prefix the plaintext with a 32-byte hash of the domain.
    # Cookie values are printable, so a non-printable first block is that prefix.
    if len(plain) > 32 and not plain[:32].isascii():
        plain = plain[32:]
    return plain.decode("utf-8", "replace")


def read_cookies() -> list[tuple[str, str, str, str]]:
    """Every `*.polar.com` cookie Chrome holds, as (host, name, value, path).

    The store is copied before reading: Chrome holds a lock on the live file, and
    a personal tool has no business writing to the browser's database anyway.
    """
    key = _keychain_key()
    source = cookie_db_path()
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "Cookies"
        shutil.copy(source, copy)
        con = sqlite3.connect(f"file:{copy}?immutable=1", uri=True)
        try:
            rows = con.execute(
                "SELECT host_key, name, encrypted_value, path FROM cookies").fetchall()
        finally:
            con.close()

    out = []
    for host, name, blob, path in rows:
        if not host.lstrip(".").endswith(_ALLOWED_HOST_SUFFIX):
            continue                          # never decrypt anything else
        value = _decrypt(blob, key)
        if value:
            out.append((host, name, value, path or "/"))
    if not out:
        raise BrowserAuthError(
            "Chrome holds no polar.com cookies. Sign in at flow.polar.com first.")
    return out


# --- renewing the short-lived token ----------------------------------------
def _jar(cookies: list[tuple[str, str, str, str]]) -> http.cookiejar.CookieJar:
    jar = http.cookiejar.CookieJar()
    expires = int(time.time()) + 86400
    for host, name, value, path in cookies:
        jar.set_cookie(http.cookiejar.Cookie(
            version=0, name=name, value=value, port=None, port_specified=False,
            domain=host, domain_specified=host.startswith("."),
            domain_initial_dot=host.startswith("."), path=path, path_specified=True,
            secure=True, expires=expires, discard=False, comment=None,
            comment_url=None, rest={}))
    return jar


def _header(jar: http.cookiejar.CookieJar) -> str:
    """A `Cookie:` header for flow.polar.com, from whatever the jar now holds."""
    parts, seen = [], set()
    for c in jar:
        if not c.domain.lstrip(".").endswith("flow.polar.com"):
            continue
        if c.name in seen:
            continue                          # a refreshed cookie shadows the old one
        seen.add(c.name)
        parts.append(f"{c.name}={c.value}")
    return "; ".join(parts)


def _to_cookies(jar: http.cookiejar.CookieJar) -> list[tuple[str, str, str, str]]:
    """The jar back in the shape `read_cookies` produces, for persisting."""
    return [(c.domain, c.name, c.value, c.path) for c in jar]


# --- keeping our own copy of the rotating session --------------------------
# Minting rotates `session_id`: the server hands the caller a new one and forgets
# the old. Whoever minted last holds the live session, so the tool has to keep its
# own copy rather than re-reading a value from Chrome that its own last run
# invalidated. `remember-me` is NOT rotated, which is why doing this cannot log
# the browser out — Chrome simply mints its own session next time it needs one.
SESSION_STORE = Path.home() / ".polarswim" / "session.json"


def _load_store() -> list[tuple[str, str, str, str]] | None:
    try:
        import json
        return [tuple(row) for row in json.loads(SESSION_STORE.read_text())]
    except (OSError, ValueError):
        return None


def _save_store(cookies: list[tuple[str, str, str, str]]) -> None:
    """Persist the rotated session, readable only by this user.

    Best-effort: a tool that cannot cache is slower, not broken, so a failure
    here must never take down a sync that has already authenticated.
    """
    import json
    try:
        SESSION_STORE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_STORE.touch(mode=0o600, exist_ok=True)
        SESSION_STORE.write_text(json.dumps(cookies))
    except OSError:
        pass


def renew(cookies: list[tuple[str, str, str, str]] | None = None) -> str:
    """Mint a fresh session by replaying the browser's own re-authentication."""
    header, updated = _mint(cookies if cookies is not None else read_cookies())
    _save_store(updated)
    return header


def _mint(cookies: list[tuple[str, str, str, str]]) -> tuple[str, list]:
    """Walk the SSO redirect chain and return the header plus the rotated jar.

    Following `/flowSso/login` with a live session present walks the OAuth
    authorization-code exchange and lands back on Flow with a new `FLOW_SESSION`
    set. Redirects are followed by the default opener; the jar collects what each
    hop sets, and the last hop is the one that matters.
    """
    jar = _jar(cookies)
    # Drop any existing token so the chain is forced to mint one. Carrying a
    # spent token shortcuts the chain to an error page instead of a renewal.
    for cookie in list(jar):
        if cookie.name == "FLOW_SESSION":
            jar.clear(cookie.domain, cookie.path, cookie.name)

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", _UA)]
    try:
        with opener.open(SSO_ENTRY, timeout=30) as resp:
            landed = resp.geturl()
    except OSError as e:
        raise BrowserAuthError(f"Polar refused the renewal: {e}")

    header = _header(jar)
    if "FLOW_SESSION=" not in header:
        raise BrowserAuthError(
            f"Polar did not issue a session; the chain ended at {landed}")
    return header, _to_cookies(jar)


def _usable(header: str) -> bool:
    """Is there a token here with enough life left to finish a sync?"""
    if "FLOW_SESSION=" not in header:
        return False
    exp = session_expiry(header)
    return bool(exp and exp - time.time() > MIN_REMAINING_S)


def credential() -> str:
    """A usable Flow cookie header, renewed if the stored one is spent.

    Two sources are tried in order, and each is given two chances — use its token
    if still live, otherwise mint a new one from it. Our own store comes first
    because it holds the rotated session this tool last minted; Chrome is the
    bootstrap, and the recovery when our copy goes stale because the browser
    minted in the meantime.
    """
    problems = []
    for name, source in (("saved session", _load_store), ("Chrome", read_cookies)):
        try:
            cookies = source()
        except BrowserAuthError as e:
            problems.append(f"{name}: {e}")
            continue
        if not cookies:
            continue
        header = _header(_jar(cookies))
        if _usable(header):
            return header
        try:
            header, updated = _mint(cookies)
            _save_store(updated)
            return header
        except BrowserAuthError as e:
            problems.append(f"{name}: {e}")

    detail = "; ".join(problems) if problems else "no session found"
    raise BrowserAuthError(
        "Could not get a session from the browser. Open flow.polar.com in Chrome, "
        "sign in if it asks, and try again.\n  (" + detail + ")")
