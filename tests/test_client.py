"""Client logic that needs no network: date windowing and id extraction."""

from datetime import date

from polarswim.client import MAX_WINDOW_DAYS, FlowClient


def _client():
    return FlowClient(cookie="FLOW_SESSION=x")


def test_windows_respect_the_api_limit():
    """Flow rejects ranges over 100 days, so every window must stay under it."""
    windows = list(_client().calendar_windows(date(2018, 1, 1), date(2026, 8, 27)))
    assert windows
    assert all((b - a).days <= MAX_WINDOW_DAYS for a, b in windows)


def test_windows_cover_the_range_without_gaps():
    start, end = date(2024, 1, 1), date(2025, 6, 15)
    windows = list(_client().calendar_windows(start, end))
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert prev_end == next_start          # contiguous, no missed days


def test_short_range_is_a_single_window():
    windows = list(_client().calendar_windows(date(2026, 1, 1), date(2026, 1, 20)))
    assert len(windows) == 1


def test_exercise_ids_dedupes_and_sorts(monkeypatch):
    """Adjacent windows share an endpoint, so the same session can appear twice."""
    events = [
        {"type": "EXERCISE", "listItemId": 2, "datetime": "2026-02-01T10:00:00.000Z"},
        {"type": "EXERCISE", "listItemId": 1, "datetime": "2026-01-01T10:00:00.000Z"},
        {"type": "EXERCISE", "listItemId": 2, "datetime": "2026-02-01T10:00:00.000Z"},
        {"type": "TRAINING_TARGET", "listItemId": 99, "datetime": "2026-03-01T00:00:00Z"},
    ]
    c = _client()
    monkeypatch.setattr(c, "calendar_events", lambda *a, **k: events)
    assert c.exercise_ids(date(2026, 1, 1), date(2026, 3, 1)) == [1, 2]
