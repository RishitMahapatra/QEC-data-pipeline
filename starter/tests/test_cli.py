"""Tests for `cli run`: the one-command Part I runner."""

from __future__ import annotations

import json

from quantum_lake_student import cli
from quantum_lake_student.models import StageResult


def _fake_stages(calls: list, fail_at: str | None = None):
    def make(name):
        def run(run_id):
            calls.append((name, run_id))
            if name == fail_at:
                raise ValueError("faail")
            return StageResult(stage=name, run_id=run_id, input_count=1, output_count=1)
        return run
    names = ["register_sources", "prepare_data", "load_postgres", "build_ml_tables", "trace_examples"]
    return lambda: [(name, make(name)) for name in names]


def test_run_calls_every_stage_in_order_with_one_run_id(monkeypatch, tmp_path) -> None:
    calls: list = []
    monkeypatch.setattr(cli, "part1_stages", _fake_stages(calls))
    monkeypatch.setattr(cli, "RESULTS_PART1", tmp_path)

    assert cli.command_run(None) == 0
    assert [name for name, _ in calls] == [
        "register_sources", "prepare_data", "load_postgres", "build_ml_tables", "trace_examples",
    ]
    assert len({run_id for _, run_id in calls}) == 1

    pipeline = json.loads((tmp_path / "run.json").read_text())["pipeline"]
    assert pipeline["run_id"] == calls[0][1]
    assert [s["stage"] for s in pipeline["stages"]] == [name for name, _ in calls]


def test_run_stops_at_the_first_failing_stage(monkeypatch, tmp_path) -> None:
    calls: list = []
    monkeypatch.setattr(cli, "part1_stages", _fake_stages(calls, fail_at="load_postgres"))
    monkeypatch.setattr(cli, "RESULTS_PART1", tmp_path)

    assert cli.command_run(None) == 1
    assert [name for name, _ in calls] == ["register_sources", "prepare_data", "load_postgres"]
    assert not (tmp_path / "run.json").exists()  # no "pipeline" entry for a failed run


def test_part1_stages_are_in_dependency_order() -> None:
    assert [name for name, _ in cli.part1_stages()] == [
        "register_sources", "prepare_data", "load_postgres", "build_ml_tables", "trace_examples",
    ]
