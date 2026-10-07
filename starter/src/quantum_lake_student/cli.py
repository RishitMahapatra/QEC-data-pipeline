"""Command-line entry point for the student workspace."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime

from rich.console import Console
from rich.table import Table

from .config import Settings
from .connections import bronze_inventory, check_platform
from .models import RESULTS_PART1, UTC, code_revision, merge_json_file


console = Console()


def command_check(settings: Settings) -> int:
    result = check_platform(settings)
    for service, message in result.items():
        console.print(f"[green]OK[/green] {service}: {message}")
    return 0


def command_inventory(settings: Settings) -> int:
    table = Table(title="Supplied course data files (kept unchanged)")
    table.add_column("Stored path")
    table.add_column("Bytes", justify="right")
    for key, size in bronze_inventory(settings):
        table.add_row(key, f"{size:,}")
    console.print(table)
    return 0


def part1_stages() -> list[tuple[str, Callable[[str], object]]]:
    """Part I stages in dependency order: Bronze -> Silver -> Gold -> ML -> trace.

    Imported here, not at module level, so `check`/`inventory` stay fast.
    """
    from .stages import build_ml_tables, load_postgres, prepare_data, register_sources, trace_examples

    return [
        ("register_sources", register_sources.run),  # verify unchanged Bronze
        ("prepare_data", prepare_data.run),          # Bronze -> six Silver tables + issues/trace
        ("load_postgres", load_postgres.run),        # Silver -> Gold (all-or-nothing) + views
        ("build_ml_tables", build_ml_tables.run),    # Gold -> two ML tables + analysis results
        ("trace_examples", trace_examples.run),      # ML -> Gold -> Silver -> Bronze proof
    ]


def command_run(_: Settings) -> int:
    """Run all of Part I with one command. Every stage gets the same run_id, so
    their run.json entries belong together; the first failing stage stops the run."""
    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC)
    summary = Table(title=f"Part I run {run_id}")
    for column, justify in (("Stage", "left"), ("Input", "right"), ("Output", "right"),
                            ("Issues", "right"), ("Seconds", "right")):
        summary.add_column(column, justify=justify)

    completed = []
    for name, stage_run in part1_stages():
        console.print(f"[cyan]>[/cyan] {name} ...")
        stage_started = time.perf_counter()
        try:
            result = stage_run(run_id)
        except Exception as exc:  # report which stage failed, then stop
            console.print(summary)
            console.print(f"[red]FAILED[/red] {name}: {type(exc).__name__}: {exc}")
            return 1
        seconds = time.perf_counter() - stage_started
        summary.add_row(name, f"{result.input_count:,}", f"{result.output_count:,}",
                        f"{result.issue_count:,}", f"{seconds:.1f}")
        completed.append({"stage": name, "seconds": round(seconds, 1)})

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "pipeline",
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "stages": completed,
        },
    )
    console.print(summary)
    console.print(f"[green]OK[/green] Part I finished; evidence in {RESULTS_PART1}/")
    return 0


def command_train(_: Settings) -> int:
    """This function runs Part II and prints the test results."""
    from .stages import train

    try:
        result = train.run(str(uuid.uuid4()))
    except Exception as exc:
        console.print(f"[red]FAILED[/red] train: {type(exc).__name__}: {exc}")
        return 1

    metrics = json.loads((train.RESULTS_PART2 / "metrics.json").read_text())
    summary = Table(title=f"Part II run {result.run_id} (test split)")
    for column, justify in (("Model", "left"), ("Logical-error rate", "right"),
                            ("Balanced acc.", "right"), ("Brier", "right")):
        summary.add_column(column, justify=justify)
    # Task B has one block of models per distance.
    blocks = [block for task in metrics.values()
              for block in ([task] if "models" in task else task["distances"].values())]
    for block in blocks:
        for model_id, m in block["models"].items():
            brier = "n/a" if m["brier_score"] is None else f"{m['brier_score']:.4f}"
            summary.add_row(model_id, f"{m['logical_error_rate']:.4f}",
                            f"{m['balanced_accuracy']:.4f}", brier)
    console.print(summary)
    console.print(f"[green]OK[/green] Part II finished; results in {train.RESULTS_PART2}/")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("check", "inventory", "run", "train"),
        help="Action to perform",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    settings = Settings.from_environment()
    commands = {
        "check": command_check,
        "inventory": command_inventory,
        "run": command_run,
        "train": command_train,
    }
    raise SystemExit(commands[arguments.command](settings))


if __name__ == "__main__":
    main()
