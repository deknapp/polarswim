"""HTTP client for Polar Flow's private web API.

Two endpoints carry everything we need:

    GET /training/getCalendarEvents?start=DD.MM.YYYY&end=DD.MM.YYYY
        Lists training sessions. Rejects windows longer than 100 days, so any
        real date range has to be walked in chunks.

    GET /api/training/analysis/{id}/details
        The full session: summary, heart-rate sample array, and — for pool swims —
        `swimDatas`, which holds the per-length records the file exports omit.

This is an undocumented internal API. It can change without notice, so responses
are validated at the parse layer and failures are surfaced loudly rather than
silently producing empty results.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator

BASE = "https://flow.polar.com"

# Flow rejects windows over 100 days; leave headroom for inclusive-endpoint quirks.
MAX_WINDOW_DAYS = 95

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")


class FlowError(RuntimeError):
    """A request failed in a way retrying will not fix."""


class SessionExpired(FlowError):
    """Flow rejected the credential (401/403, or an HTML login page)."""


@dataclass
class ClientConfig:
    min_interval_s: float = 0.4     # polite floor between requests
    max_retries: int = 3
    timeout_s: float = 30.0


class FlowClient:
    def __init__(self, cookie: str, config: ClientConfig | None = None) -> None:
        self._cookie = cookie
        self.cfg = config or ClientConfig()
        self._last_request = 0.0

    # --- plumbing -----------------------------------------------------------
    def _throttle(self) -> None:
        wait = self.cfg.min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, path: str, referer: str = f"{BASE}/diary") -> bytes:
        url = path if path.startswith("http") else BASE + path
        req = urllib.request.Request(url, headers={
            "Cookie": self._cookie,
            "Accept": "*/*",
            "Accept-Encoding": "gzip",
            "Referer": referer,
            "User-Agent": _UA,
            "X-Requested-With": "XMLHttpRequest",
        })

        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            self._throttle()
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
                    body = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise SessionExpired(
                        f"Flow rejected the session ({e.code}) on {url}. Copy a fresh "
                        "cookie from the browser."
                    ) from e
                if e.code == 400:
                    raise FlowError(f"{e.code} on {url}: {e.read()[:200]!r}") from e
                last = e                      # 5xx / 429: worth retrying
            except urllib.error.URLError as e:
                last = e
            time.sleep(1.5 * (attempt + 1))   # linear backoff
        raise FlowError(f"{url} failed after {self.cfg.max_retries} attempts: {last}")

    def _get_json(self, path: str, referer: str = f"{BASE}/diary"):
        raw = self._get(path, referer)
        text = raw.decode("utf-8", errors="replace").lstrip()
        if text.startswith("<"):
            # Flow serves the login page as HTML with a 200 when a session lapses.
            raise SessionExpired(f"{path} returned HTML, not JSON — session likely expired.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise FlowError(f"{path} returned unparseable JSON: {text[:200]!r}") from e

    # --- endpoints ----------------------------------------------------------
    def calendar_windows(self, start: date, end: date) -> Iterator[tuple[date, date]]:
        """Split [start, end] into windows Flow will accept."""
        cur = start
        while cur < end:
            stop = min(cur + timedelta(days=MAX_WINDOW_DAYS), end)
            yield cur, stop
            cur = stop

    def calendar_events(self, start: date, end: date) -> list[dict]:
        """All calendar entries in a range, walking the 100-day limit transparently."""
        events: list[dict] = []
        for a, b in self.calendar_windows(start, end):
            path = (f"/training/getCalendarEvents"
                    f"?start={a.strftime('%d.%m.%Y')}&end={b.strftime('%d.%m.%Y')}")
            chunk = self._get_json(path)
            if isinstance(chunk, list):
                events.extend(chunk)
        return events

    def exercise_ids(self, start: date, end: date) -> list[int]:
        """Deduplicated training ids in a range, oldest first.

        Windows share endpoints, so the same session can appear twice.
        """
        seen: dict[int, str] = {}
        for e in self.calendar_events(start, end):
            if e.get("type") == "EXERCISE" and e.get("listItemId"):
                seen[int(e["listItemId"])] = e.get("datetime", "")
        return [i for i, _ in sorted(seen.items(), key=lambda kv: kv[1])]

    def analysis_details(self, training_id: int) -> dict:
        """Full detail payload for one training session."""
        return self._get_json(
            f"/api/training/analysis/{training_id}/details",
            referer=f"{BASE}/training/analysis/{training_id}",
        )
