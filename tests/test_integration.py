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
