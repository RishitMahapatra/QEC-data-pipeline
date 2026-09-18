"""Bronze-only verification and registration of the three supplied QEC objects.

This module covers exactly the Bronze responsibilities the assignment
describes (``ASSIGNMENT_SPEC.md`` section 5, and this package's
``register_sources.py`` stub docstring):

- verify each object's SHA-256 hash and byte size against the course's
  release manifest (``metadata/bundle-manifest.json``);
- list every archive member path safely, without extracting or altering
  any bytes, and check the archive's own CRC integrity;
- report source objects that are missing (declared but absent) or
  unexpected (present but undeclared);
- produce a stable, content-derived registration record per object, so a
  second run on unchanged input is safe: no duplicate or drifting records.

Out of scope, on purpose: this module does not parse CSV rows, Stim
``b8``/``01`` files, or QASM circuits, and does not decide row meaning or
build Silver tables. That is ``prepare_data.py``'s job.

Run it directly::

    python -m quantum_lake_student.stages.verify_bronze_sources

By default it looks for the course bundle (a directory containing ``raw/``
and ``metadata/``) at ``/course-data`` (the Docker workspace mount) or,
failing that, by walking up from this file to find
``datasets/student-bundle/core``. Override with ``--bundle-root`` or the
``QEC_BUNDLE_ROOT`` environment variable. Results are written to
``results/part1/data_issues.parquet`` and
``results/part1/bronze_inventory.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_lake_student.models import QualityFinding, Severity, StageResult


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def is_safe_member_path(name: str) -> bool:
    """Reject archive member paths that could escape an extraction root.

    No absolute paths, no Windows drive letters, and no ``..`` traversal
    segment. Bronze must be able to *list* archive members safely without
    ever extracting something outside its own directory.
    """
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", name):
        return False
    return ".." not in Path(name.replace("\\", "/")).parts


def _finding(
    *,
    rule_id: str,
    severity: Severity,
    source_system: str,
    source_record_locator: str,
    message: str,
    observed_value: Any = None,
) -> QualityFinding:
    return QualityFinding(
        rule_id=rule_id,
        severity=severity,
        source_system=source_system,
        source_record_locator=source_record_locator,
        message=message,
        observed_value=None if observed_value is None else str(observed_value),
    )


# --------------------------------------------------------------------------
# Release manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestObject:
    """One object's declared identity, as recorded by the course release."""

    path: str  # bundle-relative, e.g. "raw/source=qec_syndromes/syndromes_dataset.zip"
    source: str
    bytes: int
    sha256: str
    mandatory: bool


def load_release_manifest(metadata_dir: Path) -> list[ManifestObject]:
    """Load the course-supplied object manifest (hashes, sizes, mandatory flags)."""
    manifest = json.loads((metadata_dir / "bundle-manifest.json").read_text(encoding="utf-8"))
    return [
        ManifestObject(
            path=entry["path"],
            source=entry["source"],
            bytes=entry["bytes"],
            sha256=entry["sha256"],
            mandatory=entry.get("mandatory", True),
        )
        for entry in manifest["objects"]
    ]


# --------------------------------------------------------------------------
# Hashing, archive integrity, and safe listing
# --------------------------------------------------------------------------


def sha256_of_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream-hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_archive_members(path: Path) -> list[dict[str, Any]]:
    """List every member path/size without extracting any bytes."""
    with zipfile.ZipFile(path) as archive:
        return [
            {"name": info.filename, "bytes": info.file_size, "is_dir": info.is_dir()}
            for info in archive.infolist()
        ]


def check_archive_integrity(path: Path) -> str | None:
    """Run the zip's own CRC check; return the first corrupt member, or None."""
    with zipfile.ZipFile(path) as archive:
        return archive.testzip()


# --------------------------------------------------------------------------
# Per-object verification and stable registration records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BronzeObjectRecord:
    """Stable, content-derived registration record for one Bronze object.

    Deliberately excludes anything time-dependent (no timestamps, no run
    ID) so a second run on unchanged input produces a byte-identical
    record -- the "no duplicate source records" / safe-rerun requirement.
    Run-level evidence (when this ran) belongs in ``run.json``/
    ``StageResult``, not here.
    """

    source: str
    object_path: str
    bytes: int
    sha256: str
    member_count: int
    mandatory: bool


@dataclass
class BronzeVerificationResult:
    findings: list[QualityFinding] = field(default_factory=list)
    records: list[BronzeObjectRecord] = field(default_factory=list)
    objects_declared: int = 0

    def merge(self, other: "BronzeVerificationResult") -> None:
        self.findings.extend(other.findings)
        self.records.extend(other.records)
        self.objects_declared += other.objects_declared

    def summary(self) -> dict[str, Any]:
        return {
            "objects_declared": self.objects_declared,
            "objects_verified": len(self.records),
            "issue_count": len(self.findings),
            "issues_by_severity": dict(Counter(f.severity.value for f in self.findings)),
            "issues_by_rule": dict(Counter(f.rule_id for f in self.findings)),
        }


def verify_bronze_object(
    local_path: Path, manifest_entry: ManifestObject
) -> BronzeVerificationResult:
    """Verify one supplied object against its manifest entry.

    Checks existence, byte-for-byte size and SHA-256 match, safe archive
    member paths, and archive CRC integrity. Never rewrites, extracts, or
    otherwise alters the object -- Bronze bytes stay exactly as supplied.
    """
    result = BronzeVerificationResult()
    locator = manifest_entry.path

    if not local_path.exists():
        result.findings.append(
            _finding(
                rule_id="bronze.missing_object",
                severity=Severity.ERROR,
                source_system=manifest_entry.source,
                source_record_locator=locator,
                message="declared source object was not found",
            )
        )
        return result

    actual_bytes = local_path.stat().st_size
    if actual_bytes != manifest_entry.bytes:
        result.findings.append(
            _finding(
                rule_id="bronze.size_mismatch",
                severity=Severity.ERROR,
                source_system=manifest_entry.source,
                source_record_locator=locator,
                message="object byte size does not match the release manifest",
                observed_value={"actual": actual_bytes, "expected": manifest_entry.bytes},
            )
        )

    actual_sha256 = sha256_of_file(local_path)
    if actual_sha256 != manifest_entry.sha256:
        result.findings.append(
            _finding(
                rule_id="bronze.hash_mismatch",
                severity=Severity.ERROR,
                source_system=manifest_entry.source,
                source_record_locator=locator,
                message="object SHA-256 does not match the release manifest",
                observed_value={"actual": actual_sha256, "expected": manifest_entry.sha256},
            )
        )

    try:
        members = list_archive_members(local_path)
    except zipfile.BadZipFile as exc:
        result.findings.append(
            _finding(
                rule_id="bronze.corrupt_archive",
                severity=Severity.ERROR,
                source_system=manifest_entry.source,
                source_record_locator=locator,
                message=str(exc),
            )
        )
        return result

    unsafe = [
        member["name"]
        for member in members
        if not member["is_dir"] and not is_safe_member_path(member["name"])
    ]
    if unsafe:
        result.findings.append(
            _finding(
                rule_id="bronze.unsafe_member_path",
                severity=Severity.ERROR,
                source_system=manifest_entry.source,
                source_record_locator=locator,
                message="archive contains member path(s) unsafe to extract",
                observed_value=unsafe,
            )
        )

    bad_member = check_archive_integrity(local_path)
    if bad_member is not None:
        result.findings.append(
            _finding(
                rule_id="bronze.corrupt_member",
                severity=Severity.ERROR,
                source_system=manifest_entry.source,
                source_record_locator=f"{locator}!{bad_member}",
                message="archive member failed its CRC check",
            )
        )

    if not any(f.severity is Severity.ERROR for f in result.findings):
        result.records.append(
            BronzeObjectRecord(
                source=manifest_entry.source,
                object_path=manifest_entry.path,
                bytes=actual_bytes,
                sha256=actual_sha256,
                member_count=sum(1 for member in members if not member["is_dir"]),
                mandatory=manifest_entry.mandatory,
            )
        )
    return result


# --------------------------------------------------------------------------
# Missing / unexpected object inventory
# --------------------------------------------------------------------------


def discover_actual_objects(raw_root: Path) -> dict[str, Path]:
    """Every file found under the raw Bronze root, keyed by its
    manifest-relative path (``raw/<...>``)."""
    return {
        "raw/" + path.relative_to(raw_root).as_posix(): path
        for path in sorted(raw_root.rglob("*"))
        if path.is_file()
    }


def check_unexpected_objects(
    manifest: list[ManifestObject], actual: dict[str, Path]
) -> list[QualityFinding]:
    """Flag files present under the raw Bronze root that the release
    manifest does not declare (leftover local artifacts, OS junk files,
    accidental copies, etc.)."""
    declared_paths = {entry.path for entry in manifest}
    return [
        _finding(
            rule_id="bronze.unexpected_object",
            severity=Severity.WARNING,
            source_system="bronze",
            source_record_locator=path,
            message="file present under the raw Bronze root is not declared in the release manifest",
        )
        for path in sorted(actual)
        if path not in declared_paths
    ]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def register_bronze_sources(bundle_root: Path) -> BronzeVerificationResult:
    """Verify and register every Bronze object under one course-bundle root
    (a directory containing ``raw/`` and ``metadata/``)."""
    manifest = load_release_manifest(bundle_root / "metadata")
    actual_objects = discover_actual_objects(bundle_root / "raw")

    result = BronzeVerificationResult(objects_declared=len(manifest))
    result.findings.extend(check_unexpected_objects(manifest, actual_objects))
    for entry in manifest:
        result.merge(verify_bronze_object(bundle_root / entry.path, entry))
    return result


def _default_bundle_root() -> Path:
    env_root = os.environ.get("QEC_BUNDLE_ROOT")
    if env_root:
        return Path(env_root)
    docker_mount = Path("/course-data")
    if docker_mount.exists():
        return docker_mount
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "datasets" / "student-bundle" / "core"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the course data bundle (needs raw/ and metadata/); "
        "pass --bundle-root explicitly or set QEC_BUNDLE_ROOT."
    )


# --------------------------------------------------------------------------
# Result evidence: results/part1/data_issues.parquet + bronze_inventory.json
# --------------------------------------------------------------------------

_ISSUE_FIELDS = [
    "rule_id",
    "severity",
    "source_system",
    "source_record_locator",
    "message",
    "observed_value",
]


def _findings_to_rows(findings: list[QualityFinding]) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": finding.rule_id,
            "severity": finding.severity.value,
            "source_system": finding.source_system,
            "source_record_locator": finding.source_record_locator,
            "message": finding.message,
            "observed_value": finding.observed_value,
        }
        for finding in findings
    ]


def write_data_issues(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Write ``data_issues.parquet``.

    Falls back to a same-named ``.csv`` file if pandas/pyarrow are not
    installed in the current interpreter (they are declared dependencies
    of this project and available in the course's Docker image).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows, columns=_ISSUE_FIELDS).to_parquet(output_path, index=False)
        return output_path
    except ImportError:
        fallback = output_path.with_suffix(".csv")
        with fallback.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_ISSUE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return fallback


def write_bronze_inventory(result: BronzeVerificationResult, output_path: Path) -> Path:
    """Write the stable per-object registration records plus a summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": result.summary(),
        "objects": [
            {
                "source": record.source,
                "object_path": record.object_path,
                "bytes": record.bytes,
                "sha256": record.sha256,
                "member_count": record.member_count,
                "mandatory": record.mandatory,
            }
            for record in result.records
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _execute(bundle_root: Path, out_dir: Path) -> tuple[BronzeVerificationResult, Path, Path]:
    result = register_bronze_sources(bundle_root)
    issues_path = write_data_issues(_findings_to_rows(result.findings), out_dir / "data_issues.parquet")
    inventory_path = write_bronze_inventory(result, out_dir / "bronze_inventory.json")
    return result, issues_path, inventory_path


def run(
    run_id: str, *, bundle_root: Path | None = None, results_dir: Path | None = None
) -> StageResult:
    """Stage entry point: verify and register the Bronze sources.

    Matches the ``run(run_id) -> StageResult`` convention used by the
    other stage modules in this package (see ``register_sources.py``).
    """
    stage_result = StageResult(stage="verify_bronze_sources", run_id=run_id)
    root = bundle_root or _default_bundle_root()
    out_dir = results_dir or Path(__file__).resolve().parents[3] / "results" / "part1"

    result, _issues_path, _inventory_path = _execute(root, out_dir)

    stage_result.input_count = result.objects_declared
    stage_result.output_count = len(result.records)
    stage_result.issue_count = len(result.findings)
    stage_result.finish()
    return stage_result


def _print_summary(result: BronzeVerificationResult) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Bronze source verification")
    table.add_column("Source")
    table.add_column("Object")
    table.add_column("Bytes", justify="right")
    table.add_column("SHA-256 (first 12)")
    table.add_column("Members", justify="right")
    for record in result.records:
        table.add_row(
            record.source,
            record.object_path,
            f"{record.bytes:,}",
            record.sha256[:12],
            str(record.member_count),
        )
    console.print(table)
    console.print(
        f"Declared objects: {result.objects_declared}  "
        f"Verified: {len(result.records)}  Issues: {len(result.findings)}"
    )
    by_rule = Counter(f.rule_id for f in result.findings)
    if by_rule:
        console.print(f"Issues by rule: {dict(by_rule)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Directory containing raw/ and metadata/ (defaults to the course layout)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Where to write data_issues.parquet/bronze_inventory.json (defaults to results/part1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = args.bundle_root or _default_bundle_root()
    out_dir = args.results_dir or Path(__file__).resolve().parents[3] / "results" / "part1"

    result, issues_path, inventory_path = _execute(root, out_dir)
    _print_summary(result)
    print(f"\nWrote {issues_path}")
    print(f"Wrote {inventory_path}")

    has_errors = any(f.severity is Severity.ERROR for f in result.findings)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
