"""The SVG workout graphic: geometry, escaping, and canvas-safety.

The image is rasterised to PNG in the browser through a canvas, which imposes two
hard constraints: no `foreignObject` and no external references, either of which
taints the canvas and makes the export silently fail.
"""

import re

import pytest

from polarswim import image

HEADER = {"start_time": "2026-08-19T17:11:11", "distance_m": 1394.46,
          "duration_s": 2822.0, "avg_hr": 126}
ZONES = [{"zone": f"Z{i}", "label": "x", "color": "#111111",
          "low": i * 30, "high": (i + 1) * 30} for i in range(1, 6)]
MIX = [{"stroke": "freestyle", "pct": 60.0, "yards": 900, "color": "#4aa3ff"},
       {"stroke": "backstroke", "pct": 40.0, "yards": 600, "color": "#3ddc84"}]


def _set(**kw):
    base = dict(set_id=1, reps=4, rep_yards=50, n=8, stroke="freestyle",
                confidence=0.8, pace_s=26.0, hr_cost=20.0, rest_before_s=15.0,
                rep_seconds=55.0, note="",
                hr_zone={"zone": "Z3", "label": "tempo", "color": "#3ddc84",
                         "pct_max": 75},
                speed={"percentile": 62, "color": "#3ddc84", "n": 400,
                       "distance": 50},
                pr=False)
    base.update(kw)
    return base


def _svg(sets=None, mix=None):
    return image.workout_svg(HEADER, sets if sets is not None else [_set()],
                             mix if mix is not None else MIX, ZONES, 172)


class TestStructure:
    def test_is_a_complete_svg_document(self):
        svg = _svg()
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg

    def test_declares_matching_width_and_viewbox(self):
        svg = _svg()
        w = int(re.search(r'width="(\d+)"', svg).group(1))
        h = int(re.search(r'height="(\d+)"', svg).group(1))
        assert f'viewBox="0 0 {w} {h}"' in svg

    def test_height_grows_with_the_number_of_sets(self):
        one = int(re.search(r'height="(\d+)"', _svg([_set()])).group(1))
        many = int(re.search(r'height="(\d+)"', _svg([_set()] * 10)).group(1))
        assert many > one

    def test_no_content_overflows_the_canvas(self):
        """An earlier layout let the legend paint over the pie."""
        svg = _svg([_set(pr=True)])
        width = int(re.search(r'width="(\d+)"', svg).group(1))
        xs = [float(m) for m in re.findall(r'[ ](?:x|cx)="([0-9.]+)"', svg)]
        assert max(xs) < width


class TestCanvasSafety:
    """Anything that taints a canvas breaks the PNG export silently."""

    def test_no_foreign_object(self):
        assert "foreignObject" not in _svg()

    def test_no_external_references(self):
        svg = _svg()
        for token in ("http://", "https://", "<image", "@import", "url("):
            if token == "http://":
                # the xmlns declaration is the one permitted occurrence
                assert svg.count(token) == svg.count('xmlns="http://')
            else:
                assert token not in svg

    def test_uses_only_generic_font_families(self):
        assert 'font-family="Helvetica,Arial,sans-serif"' in _svg()


class TestContent:
    def test_reports_the_headline_totals(self):
        svg = _svg()
        assert "2026-08-19" in svg and "1,525 yd" in svg
        assert image._fmt_time(HEADER["duration_s"]) in svg

    def test_reports_reps_not_raw_lengths(self):
        assert "4×50" in _svg()

    def test_marks_a_personal_best(self):
        assert "★ PR" in _svg([_set(pr=True)])
        assert "★ PR" not in _svg([_set(pr=False)])

    def test_flags_low_confidence_labels(self):
        assert ">?</text>" in _svg([_set(confidence=0.3)])
        assert ">?</text>" not in _svg([_set(confidence=0.9)])

    def test_states_that_stroke_is_inferred(self):
        """The graphic goes public, so it must not imply the label was measured."""
        assert "inferred" in _svg() and "not measured" in _svg()

    def test_renders_the_zone_key(self):
        svg = _svg()
        for z in ZONES:
            assert f'>{z["zone"]}</text>' in svg

    def test_missing_zone_or_speed_is_simply_omitted(self):
        svg = _svg([_set(hr_zone=None, speed=None)])
        assert svg.startswith("<svg")          # no crash, no placeholder junk


class TestPie:
    def test_multiple_strokes_draw_arcs(self):
        assert len(re.findall(r"<path d=", _svg())) == len(MIX)

    def test_a_single_stroke_draws_a_circle_not_an_arc(self):
        """A 360-degree arc is degenerate and renders as nothing."""
        solo = [{"stroke": "freestyle", "pct": 100.0, "yards": 1500,
                 "color": "#4aa3ff"}]
        svg = _svg(mix=solo)
        assert "<path d=" not in svg
        assert svg.count("<circle") >= 1

    def test_no_strokes_is_handled(self):
        assert _svg(mix=[]).startswith("<svg")


class TestEscaping:
    def test_markup_in_a_label_is_escaped(self):
        svg = image.workout_svg({**HEADER, "start_time": "<script>x</script>"},
                                [_set()], MIX, ZONES, 172)
        assert "<script>" not in svg and "&lt;script&gt;" in svg
