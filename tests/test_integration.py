"""End-to-end against the shipped sample database of real swims.

These are the tests that would catch a break in what a reviewer actually runs:
the sample database is committed, so they need no credentials and no network.
"""

import json
from pathlib import Path

import pytest

from polarswim import ai, analyze, db, render, report
from polarswim.web import create_app

SAMPLE = Path(__file__).resolve().parent.parent / "sample" / "sample.db"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample database absent")


@pytest.fixture(scope="module")
def engine():
    return db.connect(SAMPLE)


@pytest.fixture(scope="module")
def workout_id(engine):
    return int(report.workout_headers(engine).iloc[0]["id"])


def test_sample_database_has_real_swims(engine):
    s = db.summary(engine)
    assert s["pool_swims"] >= 5
    assert s["lengths"] > 300
    assert s["hr_samples"] > 10_000


def test_polar_labelled_nothing(engine):
    """The premise of the project, asserted against real data."""
    df = analyze.load_lengths(engine)
    assert set(df["polar_style"].unique()) == {"OTHER"}


def test_analysis_covers_every_length(engine):
    res = analyze.analyze(engine, persist=False)
    assert res.n_lengths == len(analyze.load_lengths(engine))
    assert set(res.counts()) <= set(analyze.CLASSES)


def test_analysis_is_deterministic(engine):
    a = analyze.analyze(engine, persist=False).counts()
    b = analyze.analyze(engine, persist=False).counts()
    assert a == b


def test_pool_length_is_recovered_when_polar_omits_it(engine):
    """Some sessions carry lengths but no poolInfo; distance still encodes it."""
    df = analyze.load_lengths(engine)
    assert df["pool_m"].notna().all() and (df["pool_m"] > 0).all()


def test_card_renders_for_a_real_workout(engine, workout_id):
    df = report.classified_lengths(engine, workout_id)
    header = report.workout_headers(engine)
    header = header[header["id"] == workout_id].iloc[0].to_dict()
    block = render.strava_block(df, header)
    assert "polarswim" in block and len(block.splitlines()) > 5


def test_season_report_over_a_date_range(engine):
    s = report.season_summary(engine)
    assert s["workouts"] >= 5 and s["yards"] > 0
    assert json.dumps(s)              # must survive the HTTP layer


def test_offline_review_needs_no_credentials(engine, workout_id):
    df = report.classified_lengths(engine, workout_id)
    res = analyze.analyze(engine, workout_id=workout_id, persist=False)
    out = ai.review_offline({}, report.sets_for_workout(df), res.params)
    assert out.model == "offline" and out.text


class TestWebUI:
    @pytest.fixture(scope="class")
    def client(self):
        return create_app(SAMPLE).test_client()

    def test_page_loads(self, client):
        r = client.get("/")
        assert r.status_code == 200 and b"polarswim" in r.data

    def test_swims_endpoint(self, client):
        r = client.get("/api/swims")
        assert r.status_code == 200 and len(r.get_json()["swims"]) >= 5

    def test_workout_endpoint(self, client, workout_id):
        d = client.get(f"/api/workout/{workout_id}").get_json()
        assert d["sets"] and d["paces"] and d["card"]
        assert len(d["paces"]) == sum(s["n"] for s in d["sets"])

    def test_report_endpoint_is_serializable(self, client):
        r = client.get("/api/report?from=2026-01-01")
        assert r.status_code == 200 and r.get_json()["text"]

    def test_unknown_workout_is_404(self, client):
        assert client.get("/api/workout/1").status_code == 404


class TestDashboardCharts:
    """The pie is drawn client-side, so assert both halves exist: the element to
    draw into, and the data to draw. A previous edit silently added the function
    without the element, and the chart rendered as nothing."""

    @pytest.fixture(scope="class")
    def client(self):
        return create_app(SAMPLE).test_client()

    def test_page_contains_the_pie_element_and_its_drawing_code(self, client):
        page = client.get("/").get_data(as_text=True)
        for token in ('id="pie"', 'id="legend"', "function drawPie", "drawPie(d.mix)"):
            assert token in page, f"dashboard is missing {token}"

    def test_workout_endpoint_supplies_pie_data(self, client, workout_id):
        mix = client.get(f"/api/workout/{workout_id}").get_json()["mix"]
        assert mix
        assert sum(m["pct"] for m in mix) == pytest.approx(100.0)
        for slice_ in mix:
            assert slice_["color"].startswith("#")
            assert slice_["yards"] > 0

    def test_sets_report_reps_not_raw_lengths(self, client, workout_id):
        sets = client.get(f"/api/workout/{workout_id}").get_json()["sets"]
        assert sets
        for s in sets:
            assert s["reps"] >= 1
            assert s["rep_yards"] >= 25
            assert s["rep_seconds"] > 0


class TestDashboardMetrics:
    """Zone, relative speed and PR columns, plus the key that explains them."""

    @pytest.fixture(scope="class")
    def client(self):
        return create_app(SAMPLE).test_client()

    def test_zone_key_is_present_and_anchored_to_the_swimmers_max(self, client, workout_id):
        d = client.get(f"/api/workout/{workout_id}").get_json()
        assert d["hr_max"] > 100
        zones = d["zones"]
        assert [z["zone"] for z in zones] == ["Z1", "Z2", "Z3", "Z4", "Z5"]
        assert zones[-1]["high"] == d["hr_max"]

    def test_sets_carry_zone_speed_and_pr_fields(self, client, workout_id):
        for s in client.get(f"/api/workout/{workout_id}").get_json()["sets"]:
            assert "hr_zone" in s and "speed" in s and "pr" in s
            assert isinstance(s["pr"], bool)
            if s["hr_zone"]:
                assert s["hr_zone"]["color"].startswith("#")
            if s["speed"]:
                assert 0 <= s["speed"]["percentile"] <= 100

    def test_page_renders_the_new_columns_and_key(self, client):
        page = client.get("/").get_data(as_text=True)
        for token in ("zone", "speed", "class=\"chip\"", "class=\"pctbar\"",
                      "class=\"pr\"", "swim maximum"):
            assert token in page, f"dashboard is missing {token}"


class TestImageExport:
    """The downloadable graphic for uploading to Strava as a photo."""

    @pytest.fixture(scope="class")
    def client(self):
        return create_app(SAMPLE).test_client()

    def test_endpoint_serves_an_svg(self, client, workout_id):
        r = client.get(f"/api/image/{workout_id}.svg")
        assert r.status_code == 200
        assert r.mimetype == "image/svg+xml"
        body = r.get_data(as_text=True)
        assert body.startswith("<svg") and body.rstrip().endswith("</svg>")

    def test_image_reflects_the_workout(self, client, workout_id):
        body = client.get(f"/api/image/{workout_id}.svg").get_data(as_text=True)
        head = client.get(f"/api/workout/{workout_id}").get_json()["header"]
        assert head["date"][:10] in body
        assert f'{head["yards"]:,} yd' in body

    def test_unknown_workout_is_404(self, client):
        assert client.get("/api/image/1.svg").status_code == 404

    def test_page_offers_the_download_button(self, client):
        page = client.get("/").get_data(as_text=True)
        assert "download image for Strava" in page
        assert "function downloadImage" in page
        assert "toBlob" in page          # the canvas rasterisation path


class TestSummaryAndPRTabs:
    @pytest.fixture(scope="class")
    def client(self):
        return create_app(SAMPLE).test_client()

    def test_summary_reports_totals_and_heart_rate(self, client):
        d = client.get("/api/summary").get_json()
        assert d["workouts"] >= 5 and d["yards"] > 0 and d["lengths"] > 0
        assert d["hr_max"] > 100 >= 0
        assert d["hr_mean"] <= d["hr_p95"] <= d["hr_max"]

    def test_summary_zone_time_covers_every_sample(self, client):
        d = client.get("/api/summary").get_json()
        assert len(d["zone_time"]) == 5
        assert sum(z["seconds"] for z in d["zone_time"]) == pytest.approx(
            d["hr_samples"], rel=0.01)
        assert sum(z["pct"] for z in d["zone_time"]) == pytest.approx(100, abs=0.5)

    def test_summary_reports_training_load(self, client):
        d = client.get("/api/summary").get_json()
        assert d["total_trimp"] > 0 and d["mean_trimp"] > 0

    def test_personal_bests_are_listed_per_distance_and_stroke(self, client):
        prs = client.get("/api/prs").get_json()["prs"]
        assert prs
        keys = [(p["yards"], p["stroke"]) for p in prs]
        assert len(keys) == len(set(keys)), "a distance/stroke pair appears twice"
        for p in prs:
            assert p["seconds"] > 0 and p["yards"] >= 25
            assert p["n_attempts"] >= 1

    def test_a_best_is_never_slower_than_its_own_pace_implies(self, client):
        for p in client.get("/api/prs").get_json()["prs"]:
            assert p["pace_per_50"] == pytest.approx(
                p["seconds"] / (p["yards"] / 50.0), rel=0.02)

    def test_workout_view_carries_both_effort_scores(self, client, workout_id):
        e = client.get(f"/api/workout/{workout_id}").get_json()["effort"]
        assert e and 0 <= e["score"] <= 100
        if e.get("intensity") is not None:
            assert 0 <= e["intensity"] <= 100

    def test_page_offers_all_three_tabs(self, client):
        page = client.get("/").get_data(as_text=True)
        for token in ('data-tab="workouts"', 'data-tab="summary"',
                      'data-tab="prs"', "function loadSummary", "function loadPRs"):
            assert token in page, f"dashboard is missing {token}"


class TestPersonalBestsPage:
    """Bests are grouped by stroke, lead with racing distances, and include medleys."""

    @pytest.fixture(scope="class")
    def client(self):
        return create_app(SAMPLE).test_client()

    def test_every_best_says_whether_its_distance_is_raced(self, client):
        prs = client.get("/api/prs").get_json()["prs"]
        assert prs and all("competitive" in p for p in prs)

    def test_racing_distances_are_a_subset_of_everything_recorded(self):
        from polarswim import metrics
        assert metrics.is_competitive(100, "freestyle")
        assert metrics.is_competitive(1650, "freestyle")
        assert not metrics.is_competitive(75, "freestyle")
        assert not metrics.is_competitive(1650, "butterfly")   # not an event

    def test_medley_distances_are_only_the_three_that_exist(self):
        from polarswim import metrics
        assert metrics.is_competitive(200, "IM")
        assert not metrics.is_competitive(50, "IM")            # no 50 IM
        assert not metrics.is_competitive(300, "IM")

    def test_the_page_offers_a_tab_for_every_stroke_and_for_medley(self, client):
        page = client.get("/").data.decode()
        assert "PR_STROKES" in page
        for stroke in ("freestyle", "backstroke", "breaststroke", "butterfly", "IM"):
            assert stroke in page

    def test_a_medley_best_carries_its_splits_and_its_form(self):
        """Built from a reference rather than the sample database, which holds no
        medleys — a data-dependent skip is a test that never runs."""
        from polarswim import db, metrics, report

        ref = metrics.SwimmerReference(hr_max=172, median_pace_s=26.0)
        ref.best_im = {100: {"seconds": 92.8, "workout_id": 1, "date": "2026-02-27",
                             "continuous": True, "splits_s": [28.8, 23.2, 24.0, 16.8],
                             "n_rounds": 6}}
        rows = report.personal_bests(db.connect(":memory:"), ref)
        medley = [r for r in rows if r["stroke"] == "IM"]
        assert len(medley) == 1
        assert medley[0]["splits_s"] == [28.8, 23.2, 24.0, 16.8]
        assert medley[0]["form"] == "continuous"
        assert medley[0]["competitive"] is True
        assert medley[0]["pace_per_50"] == pytest.approx(46.4)

    def test_a_single_stroke_best_has_no_medley_fields(self, client):
        prs = client.get("/api/prs").get_json()["prs"]
        for p in prs:
            if p["stroke"] != "IM":
                assert p["form"] is None and p["splits_s"] is None
