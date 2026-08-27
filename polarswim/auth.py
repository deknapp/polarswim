"""Session credential handling.

Flow authenticates the browser with a set of cookies, the important one being a
short-lived (~1 hour) `FLOW_SESSION` JWT. There is no documented refresh flow, so
the credential is supplied by the user: copy any authenticated request out of the
browser's network inspector as cURL, and save the cookie string.

The credential is deliberately never written into the database or the repository.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

DEFAULT_COOKIE_PATH = Path.home() / ".polarswim" / "cookie.txt"
ENV_VAR = "POLAR_COOKIE"


class AuthError(RuntimeError):
    """No usable credential, or the credential has expired."""


def _from_curl(text: str) -> str | None:
    """Extract the cookie string from a copied cURL command, if that's what we got.

    Accepts both `-b '...'` and `-H 'cookie: ...'` forms, which different browsers
    and different DevTools versions emit.
    """
    m = re.search(r"-b\s+'([^']*)'", text) or re.search(r"-b\s+\"([^\"]*)\"", text)
    if m:
        return m.group(1)
    m = re.search(r"-H\s+'cookie:\s*([^']*)'", text, re.I)
    return m.group(1) if m else None


def load_cookie(path: str | Path | None = None) -> str:
    """Resolve the cookie string from the environment or a file.

    Order: `$POLAR_COOKIE`, then the given path, then `~/.polarswim/cookie.txt`.
    A raw cURL command is accepted as-is so the user can paste without editing.
    """
    raw = os.environ.get(ENV_VAR)
    if not raw:
        p = Path(path) if path else DEFAULT_COOKIE_PATH
        if not p.exists():
            raise AuthError(
                f"No credential found. Set ${ENV_VAR}, or save your Flow cookie to {p}.\n"
                "To get it: open flow.polar.com, DevTools -> Network, filter 'api/training', "
                "right-click a request -> Copy as cURL, then `pbpaste > "
                f"{p}`."
            )
        raw = p.read_text()

    cookie = _from_curl(raw) or raw
    cookie = " ".join(cookie.split())
    if "FLOW_SESSION=" not in cookie:
        raise AuthError(
            "Credential does not contain FLOW_SESSION. It was probably copied from a "
            "request to localizations.flow.polar.com (a separate, cookie-less host). "
            "Copy a request whose URL starts with https://flow.polar.com/api/."
        )
    return cookie


def session_expiry(cookie: str) -> float | None:
    """Read the `exp` claim out of the FLOW_SESSION JWT, as a unix timestamp.

    The signature is irrelevant here — only Polar can validate it. Decoding the
    payload lets us fail with "your session expired 20 minutes ago, copy a fresh
    one" instead of an opaque 401 in the middle of a long sync.
    """
    m = re.search(r"FLOW_SESSION=([^;]+)", cookie)
    if not m:
        return None
    parts = m.group(1).split(".")
    if len(parts) < 2:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)   # restore base64 padding
    try:
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:
        return None


def assert_valid(cookie: str) -> None:
    """Raise if the session is already expired, with a human-readable margin."""
    exp = session_expiry(cookie)
    if exp is None:
        return                      # can't tell; let the request itself fail
    remaining = exp - time.time()
    if remaining <= 0:
        raise AuthError(
            f"Session expired {int(-remaining) // 60} minutes ago. "
            "Copy a fresh request from the browser and try again."
        )
