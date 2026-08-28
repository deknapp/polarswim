"""The CLI surface, and a guard against documenting flags that don't exist.

A help text or README that advertises an option the parser rejects is worse than
no documentation, so the last test here scrapes every `polarswim ...` invocation
out of the docstrings and the README and asserts the parser actually accepts it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from polarswim import cli, spark

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDERS = {
    "YYYY-MM-DD": "2026-01-01", "<date|id|latest>": "latest", "<workout_id>": "1",
    "<id>": "1", "ID": "1", "N": "1", "...": "2026-01-01", "8770": "8770",
}


def _concrete(token: str) -> str:
    return PLACEHOLDERS.get(token, token)


SUBCOMMANDS = {"sync", "status", "analyze", "card", "review", "report",
               "serve", "reparse"}


def _invocations() -> list[list[str]]:
    """Every documented `polarswim ...` command line, as argv lists.

    Help text writes a command and its description on one line separated by two
    or more spaces, so the command is everything before that gap.
    """
    sources = [ROOT / "README.md"] + sorted((ROOT / "polarswim").glob("*.py"))
    found: list[list[str]] = []
    for path in sources:
        # In Markdown only fenced code blocks are a claim that something runs;
        # naming a command inline in a sentence is not. Python docstrings are
        # all usage documentation, so every line there counts.
        in_block = path.suffix != ".md"
        for line in path.read_text().splitlines():
            if path.suffix == ".md" and line.lstrip().startswith("```"):
                in_block = not in_block
                continue
            if not in_block:
                continue
            text = line.strip().lstrip("`$ ")
            m = re.match(r"(?:python -m )?polarswim (.*)$", text)
            if not m:
                continue                       # not a command line at all
            body = m.group(1).split("`")[0]    # stop at a closing backtick
            command = re.split(r"\s{2,}", body.strip())[0]   # drop any description
            raw = command.replace("[", " ").replace("]", " ").split()
            if not raw or raw[0] not in SUBCOMMANDS:
                continue
            argv = [_concrete(t) for t in raw if _concrete(t)]
            found.append(argv)
    return found


class TestParser:
    @pytest.mark.parametrize("cmd", ["status", "reparse", "analyze", "report", "serve"])
    def test_subcommand_parses_with_no_arguments(self, cmd):
        assert cli.build_parser().parse_args([cmd]).cmd == cmd

    @pytest.mark.parametrize("cmd", ["card", "review"])
    @pytest.mark.parametrize("token", ["2026-08-19", "8432902372", "latest"])
    def test_subcommand_accepts_any_workout_reference(self, cmd, token):
        assert cli.build_parser().parse_args([cmd, token]).workout == token

    @pytest.mark.parametrize("cmd", ["card", "review"])
    def test_workout_reference_is_required(self, cmd):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([cmd])

    def test_sync_flags(self):
        a = cli.build_parser().parse_args(
            ["sync", "--from", "2026-01-01", "--to", "2026-06-01",
             "--limit", "5", "--force", "--interval", "0.5"])
        assert a.limit == 5 and a.force and a.interval == 0.5

    def test_report_date_range(self):
        a = cli.build_parser().parse_args(
            ["report", "--from", "2026-01-01", "--to", "2026-06-01", "--json"])
        assert a.date_from.year == 2026 and a.json

    def test_db_flag_accepts_a_path_or_a_url(self):
        for value in ("sample/sample.db", "postgresql+psycopg://h/polarswim"):
            assert cli.build_parser().parse_args(["--db", value, "status"]).db == value

    def test_unknown_flag_is_rejected(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["analyze", "--engine", "spark"])

    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])


class TestDocumentationMatchesReality:
    def test_some_invocations_were_found(self):
        """Guards the scraper itself — a silent zero would make this vacuous."""
        assert len(_invocations()) >= 8

    def test_every_documented_invocation_parses(self):
        """No README or help text may advertise a flag the parser rejects."""
        failures = []
        for argv in _invocations():
            try:
                cli.build_parser().parse_args(argv)
            except SystemExit:
                failures.append(" ".join(argv))
        assert not failures, f"documented but unsupported: {failures}"


class TestWorkoutResolution:
    """A date is what a person remembers; the id is what the database shows."""

    @pytest.fixture(scope="class")
    def engine(self):
        from polarswim import db
        sample = ROOT / "sample" / "sample.db"
        if not sample.exists():
            pytest.skip("sample database absent")
        return db.connect(sample)

    @pytest.fixture(scope="class")
    def known(self, engine):
        from polarswim import report
        row = report.workout_headers(engine).iloc[-1]
        return int(row["id"]), str(row["start_time"])[:10]

    def test_resolves_by_id(self, engine, known):
        wid, _ = known
        assert cli.resolve_workout(engine, str(wid)) == wid

    def test_resolves_by_date(self, engine, known):
        wid, date = known
        assert cli.resolve_workout(engine, date) == wid

    def test_latest_resolves_to_the_most_recent(self, engine, known):
        wid, _ = known
        assert cli.resolve_workout(engine, "latest") == wid
        assert cli.resolve_workout(engine, "last") == wid

    def test_unknown_date_explains_itself(self, engine):
        with pytest.raises(SystemExit, match="no pool swim found"):
            cli.resolve_workout(engine, "1999-01-01")

    def test_unknown_id_explains_itself(self, engine):
        with pytest.raises(SystemExit, match="no workout with id"):
            cli.resolve_workout(engine, "999999")

    def test_unparseable_token_suggests_the_valid_forms(self, engine):
        with pytest.raises(SystemExit, match="2026-08-19"):
            cli.resolve_workout(engine, "tuesday")


class TestReadmeClaims:
    """Numbers asserted in the README must match reality, or they will rot."""

    def test_stated_test_count_is_accurate(self, pytestconfig):
        import re as _re
        readme = (ROOT / "README.md").read_text()
        m = _re.search(r"#\s*(\d+) tests, no network", readme)
        assert m, "README no longer states a test count in the quickstart block"
        claimed = int(m.group(1))
        actual = pytestconfig.pluginmanager.getplugin("terminalreporter")._numcollected
        assert claimed == actual, (
            f"README claims {claimed} tests; the suite collects {actual}")


class TestSparkIsOptional:
    def test_importing_the_package_does_not_require_pyspark(self):
        import polarswim
        import polarswim.analyze, polarswim.cli  # noqa: F401

    def test_guard_reports_availability_without_raising(self):
        assert isinstance(spark.available(), bool)

    def test_calling_without_pyspark_explains_itself(self):
        if spark.available():
            pytest.skip("PySpark installed in this environment")
        import pandas as pd
        with pytest.raises(RuntimeError, match="requirements-spark"):
            spark.set_aggregates(pd.DataFrame())


class TestSyncRunsTheAnalysis:
    """Fetching without classifying leaves the database internally inconsistent."""

    def test_sync_accepts_the_opt_out_flag(self):
        args = cli.build_parser().parse_args(["sync", "--no-analyze"])
        assert args.no_analyze is True

    def test_analysis_is_on_by_default(self):
        assert cli.build_parser().parse_args(["sync"]).no_analyze is False


class TestSetupDocs:
    """A newcomer follows these literally, so they have to be literally right."""

    def test_the_referenced_launcher_exists_and_is_executable(self):
        import os
        launcher = ROOT / "bin" / "polarswim"
        assert launcher.is_file()
        assert os.access(launcher, os.X_OK)

    def test_every_path_the_readme_points_at_is_present(self):
        readme = (ROOT / "README.md").read_text()
        for path in ("sample/sample.db", "requirements.txt",
                     "requirements-spark.txt", "bin/polarswim",
                     "docs/workout-card.png"):
            if path in readme:
                assert (ROOT / path).exists(), f"README references missing {path}"

    def test_the_stated_minimum_python_is_not_above_what_we_run_on(self):
        import re
        import sys
        m = re.search(r"Requires Python (\d+)\.(\d+)", (ROOT / "README.md").read_text())
        assert m, "README no longer states a minimum Python version"
        assert (int(m.group(1)), int(m.group(2))) <= sys.version_info[:2]

    def test_the_symlink_step_creates_its_directory_first(self):
        """`ln -s` into ~/.local/bin fails outright when that directory does not
        exist, which it need not on a fresh machine."""
        readme = (ROOT / "README.md").read_text()
        link = readme.index("ln -s")
        assert "mkdir -p ~/.local/bin" in readme[:link]
