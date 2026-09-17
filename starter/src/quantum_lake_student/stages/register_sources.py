"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- confirm each source has the documented files (CSV set/names/header for
  qec_syndromes, companion files per experiment for google_qec, source +
  transpiled circuits for qasmbench) - a missing required file stops the run,
  same severity as an unsafe archive member;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import bronze_inventory, minio_client
from quantum_lake_student.models import StageResult

MANIFEST_PATH = Path("/course-data/metadata/bundle-manifest.json")

# Anchored to this file's location (src/quantum_lake_student/stages/) rather
# than a bare relative path, so results/part1/ always lands at the project
# root regardless of the caller's current working directory - a notebook's
# kernel cwd is its own folder (e.g. /workspace/notebooks), not /workspace.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PART1 = _PROJECT_ROOT / "results" / "part1"

# Structural expectations from assignment/data-sources.md. Checking that these
# files exist and are named as documented does not require parsing their
# content (no CSV/b8/QASM decoding), so it stays a Bronze-level check.
SYNDROME_CSV_PATTERN = re.compile(r"^d-3_pfr-\d+\.\d{6}_nb-10M\.csv$")
SYNDROME_CSV_HEADER = "labels,syndromes,quantity"

GOOGLE_REQUIRED_COMPANIONS = (
    "properties.yml",
    "measurements.b8",
    "sweep.b8",
    "detection_events.b8",
    "obs_flips_actual.01",
    "obs_flips_predicted_by_belief_matching.01",
    "obs_flips_predicted_by_correlated_matching.01",
    "obs_flips_predicted_by_pymatching.01",
    "obs_flips_predicted_by_tensor_network_contraction.01",
    "circuit_ideal.stim",
    "circuit_noisy.stim",
    "circuit_detector_error_model.dem",
    "layout.svg",
    "pij_from_even_for_odd.dem",
    "pij_from_odd_for_even.dem",
)
GOOGLE_EXPECTED_EXPERIMENT_COUNT = 5

QASMBENCH_CIRCUITS = ("error_correctiond3_n5", "qec_en_n5", "qec_sm_n5")


class BronzeError(RuntimeError):
    """A Bronze object failed verification. The caller must stop the run."""


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _read_object_bytes(settings: Settings, object_name: str) -> bytes:
    client = minio_client(settings)
    resp = client.get_object(settings.s3_bucket, object_name)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def is_unsafe_member(name: str) -> bool:
    """True if a zip member path could escape the archive directory."""
    normalized = name.replace("\\", "/")
    parts = normalized.split("/")
    return normalized.startswith("/") or ".." in parts or ":" in parts[0]


def _bronze_object_name(manifest_object: dict, inventory: dict[str, int]) -> str | None:
    """Map a manifest entry ('raw/source=X/f.zip') to its Bronze key ('bronze/source=X/f.zip')."""
    candidate = manifest_object["path"].replace("raw/", "bronze/", 1)
    return candidate if candidate in inventory else None


def _check_syndrome_structure(zf: zipfile.ZipFile, members: list[str]) -> None:
    """Every qec_syndromes CSV must exist, be named as documented, and have the documented header."""
    csv_members = sorted(m for m in members if m.endswith(".csv"))
    if len(csv_members) != 7:
        raise BronzeError(
            f"expected 7 qec_syndromes CSV files, found {len(csv_members)}: {csv_members}"
        )
    for member in csv_members:
        if not SYNDROME_CSV_PATTERN.match(member):
            raise BronzeError(
                f"qec_syndromes file name does not match the documented "
                f"'d-3_pfr-<rate>_nb-10M.csv' pattern: {member}"
            )
        header = zf.read(member).split(b"\n", 1)[0].decode("utf-8").strip()
        if header != SYNDROME_CSV_HEADER:
            raise BronzeError(
                f"{member}: header {header!r} does not match the documented "
                f"header {SYNDROME_CSV_HEADER!r}"
            )


def _check_google_structure(members: list[str]) -> None:
    """Every google_qec experiment directory must have all documented companion files."""
    experiment_dirs = sorted(
        {m.rsplit("/", 1)[0] for m in members if m.endswith("/properties.yml")}
    )
    if len(experiment_dirs) != GOOGLE_EXPECTED_EXPERIMENT_COUNT:
        raise BronzeError(
            f"expected {GOOGLE_EXPECTED_EXPERIMENT_COUNT} google_qec experiment "
            f"directories, found {len(experiment_dirs)}: {experiment_dirs}"
        )
    member_set = set(members)
    for experiment_dir in experiment_dirs:
        missing = [
            name
            for name in GOOGLE_REQUIRED_COMPANIONS
            if f"{experiment_dir}/{name}" not in member_set
        ]
        if missing:
            raise BronzeError(
                f"google_qec experiment {experiment_dir!r} is missing required "
                f"companion file(s): {missing}"
            )


def _check_qasmbench_structure(members: list[str]) -> None:
    """Every documented QASMBench circuit must have both its source and transpiled variant."""
    member_set = set(members)
    for circuit in QASMBENCH_CIRCUITS:
        for variant in (f"{circuit}.qasm", f"{circuit}_transpiled.qasm"):
            path = f"small/{circuit}/{variant}"
            if path not in member_set:
                raise BronzeError(f"qasmbench is missing required circuit file: {path}")


def _code_revision() -> str:
    """Best-effort git commit hash for the code that produced this run.

    Falls back to "unknown" outside a git checkout instead of failing the
    stage - Bronze registration should not depend on git being available.
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


#this function is used to merge the new payload into the existing JSON file without overwriting the entire file. It reads the existing JSON file, updates it with the new payload under the specified key, and writes it back to the file.
def _merge_json_file(path: Path, key: str, payload: dict) -> None:
    """Set one top-level key in a shared JSON file without disturbing the rest.

    run.json and row_counts.json describe the whole pipeline run, not just
    one stage. Each stage (register_sources, prepare_data, load_postgres,
    build_ml_tables) calls this with its own stage name as `key`, so the
    files accumulate one section per stage across separate `run()` calls
    instead of each stage overwriting what the previous one wrote.
    """
    existing: dict = json.loads(path.read_text()) if path.exists() else {}
    existing[key] = payload
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))


# this one is the main composition of the bronze stage, it coordinates the whole process of registering and verifying the original source files.
def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    result = StageResult(stage="register_sources", run_id=run_id)

    inventory = dict(bronze_inventory(settings))
    result.input_count = len(inventory)
    manifest = _load_manifest()
    mandatory = [obj for obj in manifest["objects"] if obj["mandatory"]]

    registrations: list[dict] = []
    expected_names: set[str] = set()

    for obj in mandatory:
        object_name = _bronze_object_name(obj, inventory)
        expected_names.add(obj["path"].replace("raw/", "bronze/", 1))
        if object_name is None:
            raise BronzeError(f"missing mandatory source object: {obj['path']}")

        data = _read_object_bytes(settings, object_name)
        digest = hashlib.sha256(data).hexdigest()
        if digest != obj["sha256"]:
            raise BronzeError(f"hash mismatch for {object_name}")
        if len(data) != obj["bytes"]:
            raise BronzeError(f"size mismatch for {object_name}")

        #we check the contents of the zip file to ensure that it does not contain any unsafe members and has thwe right structure
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = zf.namelist()
            unsafe = [m for m in members if is_unsafe_member(m)]
            if unsafe:
                raise BronzeError(f"unsafe archive member(s) in {object_name}: {unsafe}")

            if obj["source"] == "qec_syndromes":
                _check_syndrome_structure(zf, members)
            elif obj["source"] == "google_qec":
                _check_google_structure(members)
            elif obj["source"] == "qasmbench":
                _check_qasmbench_structure(members)

        registrations.append(
            {
                "source_name": obj["source"],
                "bronze_object": object_name,
                "input_sha256": digest,
                "size_bytes": len(data),
                "member_count": len(members),
            }
        )

    unexpected = sorted(set(inventory) - expected_names)
    if unexpected:

        # object doesn't corrupt a mandatory one.mMaybe we shoiukd do sth about it? DISCUSS
        print(f"WARNING: unexpected Bronze objects present: {unexpected}")

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        sorted(registrations, key=lambda r: r["source_name"])
    )
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
            "input_hashes": {
                r["source_name"]: r["input_sha256"] for r in registrations
            },
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
