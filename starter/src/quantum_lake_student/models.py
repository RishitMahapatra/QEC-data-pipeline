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

import os
import subprocess
from pathlib import Path
from typing import Any

RESULTS_PART1 = Path("results/part1")  # the folder where all Part I evidence files live


def code_revision() -> str:
    """Best-effort git commit hash for the code that produced this run.

    Tries `git rev-parse HEAD` first (works on a normal checkout). The course
    workspace container has no git binary and mounts only starter/, so
    compose.yaml also mounts the repository's .git read-only and sets
    REPO_GIT_DIR; the commit is then read from it directly. Falls back to
    "unknown" instead of failing the stage - no stage should depend on git.
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
        pass
    git_dir = os.environ.get("REPO_GIT_DIR")
    return revision_from_git_dir(Path(git_dir)) if git_dir else "unknown"


def revision_from_git_dir(git_dir: Path) -> str:
    """Resolve HEAD to a commit hash by reading a .git directory, without git.

    HEAD is either a commit hash (detached) or "ref: refs/heads/<branch>";
    a branch's hash is in its loose ref file, or in packed-refs after gc.
    """
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: "):]
        loose = git_dir / ref
        if loose.exists():
            return loose.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        pass
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
