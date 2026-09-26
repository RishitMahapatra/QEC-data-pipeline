"""Small neutral records; these do not prescribe a target data model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

import subprocess
from pathlib import Path
from typing import Any

RESULTS_PART1 = Path("results/part1")  # the folder where all Part I evidence files live


def code_revision() -> str:
    """Best-effort git commit hash for the code that produced this run.

    Falls back to "unknown" outside a git checkout instead of failing the
    stage - no stage should depend on git being available.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def merge_json_file(path: Path, *keys: str, payload: dict) -> None:
    """Set one nested key path in a shared JSON file without disturbing the rest.

    run.json and row_counts.json describe the whole pipeline run, not just one
    stage - other stages, and other sources within the same stage, add their
    own keys. Reading the existing file first and only replacing the given
    key path means reruns don't wipe out what else was already written.
    A single key (e.g. "register_sources") sets a top-level entry; more than
    one key (e.g. "prepare_data", "google_qec") nests through an intermediate
    dict, creating it if needed.
    """
    existing: dict = json.loads(path.read_text()) if path.exists() else {}
    node = existing
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = payload
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))


def stable_record_hash(value: Any) -> str:
    """Return the same SHA-256 hash for equivalent JSON-compatible data."""
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class QualityFinding:
    rule_id: str
    severity: Severity
    source_system: str
    source_record_locator: str
    message: str
    observed_value: str | None = None


@dataclass
class StageResult:
    stage: str
    run_id: str
    input_count: int = 0
    output_count: int = 0
    issue_count: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def finish(self) -> None:
        self.finished_at = datetime.now(UTC)
