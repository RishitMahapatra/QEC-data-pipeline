"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import bronze_inventory, minio_client
from quantum_lake_student.models import StageResult

MANIFEST_OBJECT = "metadata/course-release/bundle-manifest.json"

# Anchored to this file's location rather than a bare relative path, so
# results/part1/ always lands at the project root regardless of the caller's
# current working directory (a notebook kernel's cwd is its own folder, not
# the project root).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PART1 = _PROJECT_ROOT / "results" / "part1"

EXPECTED_QASM_BENCHMARKS = {
    "error_correctiond3_n5",
    "qec_en_n5",
    "qec_sm_n5",
}
EXPECTED_GOOGLE_EXPERIMENTS = {
    "surface_code_bX_d3_r25_center_3_5",
    "surface_code_bX_d3_r25_center_5_3",
    "surface_code_bX_d3_r25_center_5_7",
    "surface_code_bX_d3_r25_center_7_5",
    "surface_code_bX_d5_r25_center_5_5",
}
GOOGLE_REQUIRED_FILES = {
    "properties.yml", "layout.svg", "circuit_ideal.stim", "circuit_noisy.stim",
    "circuit_detector_error_model.dem", "pij_from_even_for_odd.dem",
    "pij_from_odd_for_even.dem", "measurements.b8", "sweep.b8",
    "detection_events.b8", "obs_flips_actual.01",
    "obs_flips_predicted_by_pymatching.01",
    "obs_flips_predicted_by_correlated_matching.01",
    "obs_flips_predicted_by_belief_matching.01",
    "obs_flips_predicted_by_tensor_network_contraction.01",
}
SYNDROME_HEADER = "labels,syndromes,quantity"


class BronzeVerificationError(RuntimeError):
    """Raised when a supplied source object fails a required Bronze check."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member_names(data: bytes, object_name: str) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
    unsafe = [
        name for name in names
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
    ]
    if unsafe:
        raise BronzeVerificationError(
            f"Unsafe archive member path(s) in {object_name}: {unsafe[:5]}"
        )
    return names


def _verify_syndromes(data: bytes, names: list[str]) -> None:
    csv_members = [n for n in names if n.endswith(".csv")]
    if len(csv_members) != 7:
        raise BronzeVerificationError(
            f"Expected 7 syndrome CSV files, found {len(csv_members)}: {csv_members}"
        )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in csv_members:
            header = archive.read(member).splitlines()[0].decode("utf-8")
            if header != SYNDROME_HEADER:
                raise BronzeVerificationError(f"Unexpected header in {member}: {header}")


def _verify_qasmbench(names: list[str]) -> None:
    benchmarks = {n.split("/")[1] for n in names if n.startswith("small/") and n.endswith(".qasm")}
    if benchmarks != EXPECTED_QASM_BENCHMARKS:
        raise BronzeVerificationError(f"QASMBench benchmark set mismatch: {sorted(benchmarks)}")
    for benchmark in EXPECTED_QASM_BENCHMARKS:
        source = f"small/{benchmark}/{benchmark}.qasm"
        transpiled = f"small/{benchmark}/{benchmark}_transpiled.qasm"
        missing = [p for p in (source, transpiled) if p not in names]
        if missing:
            raise BronzeVerificationError(f"QASMBench {benchmark} missing: {missing}")


def _verify_google(names: list[str]) -> None:
    experiments = {n.split("/")[0] for n in names if "/" in n and n.startswith("surface_code_")}
    if experiments != EXPECTED_GOOGLE_EXPERIMENTS:
        raise BronzeVerificationError(f"Google experiment set mismatch: {sorted(experiments)}")
    name_set = set(names)
    for experiment in experiments:
        missing = {f for f in GOOGLE_REQUIRED_FILES if f"{experiment}/{f}" not in name_set}
        if missing:
            raise BronzeVerificationError(
                f"Google experiment {experiment} missing files: {sorted(missing)}"
            )


_SOURCES = (
    (
        "raw/source=qasmbench/qasmbench-qec.zip",
        "bronze/source=qasmbench/qasmbench-qec.zip",
        lambda data, names: _verify_qasmbench(names),
    ),
    (
        "raw/source=qec_syndromes/syndromes_dataset.zip",
        "bronze/source=qec_syndromes/syndromes_dataset.zip",
        lambda data, names: _verify_syndromes(data, names),
    ),
    (
        "raw/source=google_qec/google-surface-code-curated.zip",
        "bronze/source=google_qec/google-surface-code-curated.zip",
        lambda data, names: _verify_google(names),
    ),
)


def _code_revision() -> str:
    #Identify which commit produced this run's registrations, for run.json.
    try:
        completed = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return "git-unavailable"
    return completed.stdout.strip() if completed.returncode == 0 else "no-git-checkout"


def _merge_json_file(path: Path, key: str, payload: dict) -> None:
    #Overwrite this stage's section of a JSON file shared across the pipeline.
    existing: dict = json.loads(path.read_text()) if path.exists() else {}
    existing[key] = payload
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))


def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    client = minio_client(settings)
    result = StageResult(stage="register_sources", run_id=run_id)

    inventory = dict(bronze_inventory(settings))
    result.input_count = len(inventory)

    manifest = json.loads(client.get_object(settings.s3_bucket, MANIFEST_OBJECT).read())
    expected_by_path = {item["path"]: item for item in manifest["objects"]}

    registrations: list[dict] = []
    expected_bronze_names: set[str] = set()

    for manifest_path, object_name, verify in _SOURCES:
        expected = expected_by_path[manifest_path]
        expected_bronze_names.add(object_name)

        if object_name not in inventory:
            raise BronzeVerificationError(f"missing mandatory source object: {object_name}")

        data = client.get_object(settings.s3_bucket, object_name).read()

        if len(data) != expected["bytes"]:
            raise BronzeVerificationError(
                f"{object_name}: size mismatch, expected {expected['bytes']}, got {len(data)}"
            )
        actual_hash = _sha256(data)
        if actual_hash != expected["sha256"]:
            raise BronzeVerificationError(
                f"{object_name}: checksum mismatch, expected {expected['sha256']}, got {actual_hash}"
            )

        names = _safe_member_names(data, object_name)
        verify(data, names)

        registrations.append(
            {
                "source_name": expected["source"],
                "bronze_object": object_name,
                "input_sha256": actual_hash,
                "size_bytes": len(data),
                "member_count": len(names),
            }
        )

    unexpected = sorted(set(inventory) - expected_bronze_names)
    if unexpected:
        print(f"WARNING: unexpected Bronze objects present: {unexpected}")

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    # Overwritten wholesale on every run (sorted for a stable diff), not
    # appended to - a second run reflects the current Bronze state exactly
    # once, instead of accumulating one row per historical run.
    table = pa.Table.from_pylist(sorted(registrations, key=lambda r: r["source_name"]))
    pq.write_table(table, RESULTS_PART1 / "source_registration.parquet")

    result.output_count = len(registrations)
    result.issue_count = len(unexpected)
    result.finish()

    _merge_json_file(
        RESULTS_PART1 / "run.json",
        key="register_sources",
        payload={
            "run_id": run_id,
            "code_revision": _code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_count": result.input_count,
            "output_count": result.output_count,
            "issue_count": result.issue_count,
            "input_hashes": {r["source_name"]: r["input_sha256"] for r in registrations},
        },
    )
    _merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        key="register_sources",
        payload={
            "read": result.input_count,
            "accepted": result.output_count,
            "rejected": result.issue_count,
            "objects": {
                r["source_name"]: {
                    "member_count": r["member_count"],
                    "size_bytes": r["size_bytes"],
                }
                for r in registrations
            },
        },
    )

    return result
