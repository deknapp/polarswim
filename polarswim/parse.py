"""Turn Flow's detail payload into flat rows.

Flow nests everything under the exercise id and expresses every duration as an
ISO-8601 period string (`PT29.6S`, `PT47M2.875S`). Heart rate arrives as a bare
array plus a sampling interval rather than timestamped points. This module is pure
— no I/O — so the awkward parts are unit-testable against a saved payload.

A session can contain more than one exercise (multisport), so parsing yields a
list rather than assuming one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ISO_DUR = re.compile(
    r"^P(?:(?P<days>[\d.]+)D)?"
    r"(?:T(?:(?P<hours>[\d.]+)H)?(?:(?P<minutes>[\d.]+)M)?(?:(?P<seconds>[\d.]+)S)?)?$"
)


class ParseError(ValueError):
    """The payload did not have the shape we require."""


def iso_duration_seconds(value: str | None) -> float | None:
    """`PT47M2.875S` -> 2822.875. Returns None for null/empty input.

    Flow only ever emits day/hour/minute/second components; anything longer is a
    signal the format changed and we should fail rather than guess.
    """
    if value in (None, ""):
        return None
    m = _ISO_DUR.match(value)
    if not m:
        raise ParseError(f"unrecognized ISO-8601 duration: {value!r}")
    parts = m.groupdict()
    return (float(parts["days"] or 0) * 86400
            + float(parts["hours"] or 0) * 3600
            + float(parts["minutes"] or 0) * 60
            + float(parts["seconds"] or 0))


@dataclass
class Length:
    """One pool length as Polar's turn detection recorded it."""
    idx: int
    start_offset_s: float
    duration_s: float
    polar_style: str | None      # near-always "OTHER" on an armband sensor
    strokes: int | None


@dataclass
class Workout:
    id: int
    start_time: str
    stop_time: str | None
    sport_parent: str | None
    sport_id: int | None
    duration_s: float | None
    distance_m: float | None
    calories: int | None
    avg_hr: int | None
    max_hr: int | None
    pool_length_m: float | None
    pool_type: str | None
    pool_lengths_reported: int | None
    hr_interval_s: float | None
    hr_values: list[int] = field(default_factory=list)
    lengths: list[Length] = field(default_factory=list)

    @property
    def is_pool_swim(self) -> bool:
        return bool(self.lengths)


def _swim_block(details: dict, ex_id: str) -> tuple[dict, dict]:
    """(swimmingSamples, swimmingStatistics) for an exercise, or two empty dicts."""
    sd = (details.get("swimDatas") or {}).get(ex_id) or {}
    return sd.get("swimmingSamples") or {}, sd.get("swimmingStatistics") or {}


def parse_details(details: dict) -> list[Workout]:
    """Extract every exercise in one `/analysis/{id}/details` payload."""
    exercises = details.get("exercises")
    if not isinstance(exercises, dict):
        raise ParseError("payload has no 'exercises' object — API shape changed?")

    out: list[Workout] = []
    for ex_id, ex in exercises.items():
        samples, stats = _swim_block(details, ex_id)
        pool_info = stats.get("poolInfo") or {}

        hr_block = (((details.get("samples") or {}).get(ex_id) or {})
                    .get("HEART_RATE") or {}).get("samples") or {}
        hr_values = [v for v in (hr_block.get("values") or []) if isinstance(v, int)]

        lengths = []
        for i, rec in enumerate(samples.get("swimmingPool") or [], start=1):
            dur = iso_duration_seconds(rec.get("duration"))
            off = iso_duration_seconds(rec.get("startOffset"))
            if dur is None or off is None:
                continue        # a length with no timing is unusable, skip it
            lengths.append(Length(
                idx=i, start_offset_s=off, duration_s=dur,
                polar_style=rec.get("style"), strokes=rec.get("strokes"),
            ))

        out.append(Workout(
            id=int(ex_id),
            start_time=ex.get("startTime"),
            stop_time=ex.get("stopTime"),
            sport_parent=ex.get("sportParent"),
            sport_id=(ex.get("sport") or {}).get("id"),
            duration_s=iso_duration_seconds(ex.get("duration")),
            distance_m=ex.get("distance"),
            calories=ex.get("calories"),
            avg_hr=(stats.get("otherSwimmingStatistics") or {}).get("heartrateAvg"),
            max_hr=(stats.get("otherSwimmingStatistics") or {}).get("heartrateMax"),
            pool_length_m=pool_info.get("poolLength"),
            pool_type=pool_info.get("poolType"),
            pool_lengths_reported=stats.get("poolSwim"),
            hr_interval_s=iso_duration_seconds(hr_block.get("interval")),
            hr_values=hr_values,
            lengths=lengths,
        ))
    return out
