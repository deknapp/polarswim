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
    "YYYY-MM-DD": "2026-01-01", "<workout_id>": "1", "<id>": "1", "ID": "1",
    "N": "1", "...": "2026-01-01", "8770": "8770",
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
    def test_subcommand_requiring_a_workout_id(self, cmd):
        assert cli.build_parser().parse_args([cmd, "42"]).workout_id == 42

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
