"""Parse, check, and connect the supplied QEC data.

Apply documented checks, normalize nested/wide data, connect valid
relationships, and write prepared Parquet tables with chosen column names and
types. Write invalid records and the reason for exclusion to a machine-readable
data-issues output. The project README defines where generated files live.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import bronze_inventory, minio_client
from quantum_lake_student.formats import b8_record_bytes, parse_01_records
from quantum_lake_student.models import Severity, StageResult, stable_record_hash
from quantum_lake_student.stages.register_sources import RESULTS_PART1, _code_revision

# ---------------------------------------------------------------------------
# google_qec Silver: experiment.parquet and shot.parquet
#
# A problem that makes a whole experiment unusable (bad metadata, name vs
# properties.yml mismatch, missing or wrong-length file, non-binary or
# misaligned 01 file) raises GoogleSilverError and stops the run. A problem
# confined to one shot (non-zero padding bits) excludes that shot and yields a
# data-issue row.
# ---------------------------------------------------------------------------

GOOGLE_SOURCE_NAME = "google_qec"
GOOGLE_DECODERS = (
    "belief_matching",
    "correlated_matching",
    "pymatching",
    "tensor_network_contraction",
)
GOOGLE_REQUIRED_PROPERTIES = (
    "basis",
    "distance",
    "rounds",
    "shots",
    "center_data_qubit_row",
    "center_data_qubit_col",
    "circuit_measurements",
    "circuit_sweep_bits",
    "circuit_detectors",
)
_GOOGLE_DIR_PATTERN = re.compile(
    r"^surface_code_b(?P<basis>[A-Za-z])_d(?P<distance>\d+)_r(?P<rounds>\d+)"
    r"_center_(?P<row>\d+)_(?P<col>\d+)$"
)

# (shot-table column, companion file, properties.yml key holding the bits per shot)
GOOGLE_B8_FILES = (
    ("measurement_bits", "measurements.b8", "circuit_measurements"),
    ("sweep_bits", "sweep.b8", "circuit_sweep_bits"),
    ("detector_bits", "detection_events.b8", "circuit_detectors"),
)

GOOGLE_EXPERIMENT_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("basis", pa.string()),
        ("distance", pa.int32()),
        ("rounds", pa.int32()),
        ("shots", pa.int64()),
        ("center_row", pa.int32()),
        ("center_col", pa.int32()),
        ("measurement_count", pa.int32()),
        ("detector_count", pa.int32()),
    ]
)
GOOGLE_SHOT_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("shot_index", pa.int64()),
        ("measurement_bits", pa.binary()),
        ("sweep_bits", pa.binary()),
        ("detector_bits", pa.binary()),
        ("detector_event_count", pa.int32()),
        ("actual_observable_flip", pa.bool_()),
        *[(f"{name}_prediction", pa.bool_()) for name in GOOGLE_DECODERS],
    ]
)
TRACE_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("source_name", pa.string()),
        ("bronze_object", pa.string()),
        ("archive_member", pa.string()),
        ("record_locator", pa.string()),
        ("input_sha256", pa.string()),
    ]
)
DATA_ISSUES_SCHEMA = pa.schema(
    [
        ("issue_id", pa.string()),
        ("run_id", pa.string()),
        ("source_record_id", pa.string()),
        ("rule_id", pa.string()),
        ("severity", pa.string()),
        ("observed_value", pa.string()),
        ("action", pa.string()),
        ("reason", pa.string()),
    ]
)

SILVER_EXPERIMENT_PATH = "silver/google_qec/experiment.parquet"
SILVER_SHOT_PATH = "silver/google_qec/shot.parquet"


class GoogleSilverError(RuntimeError):
    """A check failed that makes an experiment unusable. The caller must stop the run."""


@dataclass
class GoogleSilver:
    experiment: pa.Table
    shot: pa.Table
    trace: pa.Table
    issues: list
    shots_read: int


def load_google_archive(settings: Settings) -> tuple[zipfile.ZipFile, str, str]:
    """Read the google_qec Bronze zip; return (archive, bronze object name, sha256 of its bytes)."""
    object_name = next(
        (name for name, _size in bronze_inventory(settings) if "source=google_qec" in name), None
    )
    if object_name is None:
        raise GoogleSilverError("no google_qec object found in Bronze")
    if settings.lake_backend == "local":
        data = (settings.local_lake_root / object_name).read_bytes()
    else:
        response = minio_client(settings).get_object(settings.s3_bucket, object_name)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
    return zipfile.ZipFile(io.BytesIO(data)), object_name, hashlib.sha256(data).hexdigest()


def _read_member(zf: zipfile.ZipFile, member: str) -> bytes:
    try:
        return zf.read(member)
    except KeyError as error:
        raise GoogleSilverError(f"required archive member is missing: {member}") from error


def google_experiment_dirs(members: list[str]) -> list[str]:
    """Experiment directories in the archive, identified by their properties.yml."""
    return sorted({m.rsplit("/", 1)[0] for m in members if m.endswith("/properties.yml")})


def read_google_properties(zf: zipfile.ZipFile, exp_dir: str) -> dict:
    props = yaml.safe_load(_read_member(zf, f"{exp_dir}/properties.yml"))
    missing = [
        key
        for key in GOOGLE_REQUIRED_PROPERTIES
        if not isinstance(props, dict) or key not in props
    ]
    if missing:
        raise GoogleSilverError(f"{exp_dir}/properties.yml is missing key(s): {missing}")
    for key in GOOGLE_REQUIRED_PROPERTIES:
        if key != "basis" and (not isinstance(props[key], int) or props[key] < 0):
            raise GoogleSilverError(f"{exp_dir}: {key}={props[key]!r} is not a non-negative integer")
    if props["shots"] <= 0:
        raise GoogleSilverError(f"{exp_dir}: shots must be positive, got {props['shots']}")
    return props


def check_google_experiment_name(exp_dir: str, props: dict) -> None:
    """The directory name and properties.yml must describe the same experiment."""
    match = _GOOGLE_DIR_PATTERN.match(exp_dir)
    if not match:
        raise GoogleSilverError(
            f"experiment directory name {exp_dir!r} does not match the documented pattern"
        )
    from_name = (
        match["basis"], int(match["distance"]), int(match["rounds"]),
        int(match["row"]), int(match["col"]),
    )
    from_props = (
        props["basis"], props["distance"], props["rounds"],
        props["center_data_qubit_row"], props["center_data_qubit_col"],
    )
    if from_name != from_props:
        raise GoogleSilverError(
            f"{exp_dir}: name says (basis, distance, rounds, row, col)={from_name} "
            f"but properties.yml says {from_props}"
        )


def check_b8_size(data: bytes, bits_per_shot: int, shots: int, label: str) -> int:
    """Return bytes per shot; raise unless the file is exactly shots x that many bytes."""
    record_bytes = 0 if bits_per_shot == 0 else b8_record_bytes(bits_per_shot)
    if len(data) != record_bytes * shots:
        raise GoogleSilverError(
            f"{label}: {len(data)} bytes, expected {record_bytes * shots} "
            f"({shots} shots x {record_bytes} bytes for {bits_per_shot} bits)"
        )
    return record_bytes

#will later be used for the data issues parquet
def dirty_padding_shots(data: bytes, bits_per_shot: int) -> list[int]:
    """Shot indices whose unused padding bits (high bits of the last byte) are not all zero."""
    pad_start = bits_per_shot % 8
    if bits_per_shot == 0 or pad_start == 0:
        return []
    record_bytes = b8_record_bytes(bits_per_shot)
    mask = (0xFF << pad_start) & 0xFF
    return [i for i, last in enumerate(data[record_bytes - 1 :: record_bytes]) if last & mask]


def read_01_flags(data: bytes, shots: int, label: str) -> list[bool]:
    """Parse a Stim 01 file: one 0/1 per line, exactly one line per shot."""
    try:
        values = parse_01_records(data)
    except ValueError as error:
        raise GoogleSilverError(f"{label}: {error}") from error
    if len(values) != shots:
        raise GoogleSilverError(f"{label}: {len(values)} lines, expected {shots}")
    return [bool(v) for v in values]


def google_experiment_source_record_id(exp_dir: str) -> str:
    return stable_record_hash(
        {"source": GOOGLE_SOURCE_NAME, "member": f"{exp_dir}/properties.yml"}
    )


def google_shot_source_record_id(exp_dir: str, shot_index: int) -> str:
    return stable_record_hash(
        {"source": GOOGLE_SOURCE_NAME, "experiment_id": exp_dir, "shot_index": shot_index}
    )


def build_google_experiment_row(exp_dir: str, props: dict) -> dict:
    return {
        "source_record_id": google_experiment_source_record_id(exp_dir),
        "experiment_id": exp_dir,
        "basis": props["basis"],
        "distance": props["distance"],
        "rounds": props["rounds"],
        "shots": props["shots"],
        "center_row": props["center_data_qubit_row"],
        "center_col": props["center_data_qubit_col"],
        "measurement_count": props["circuit_measurements"],
        "detector_count": props["circuit_detectors"],
    }


def _padding_issue(run_id: str, exp_dir: str, shot_index: int, member: str) -> dict:
    source_record_id = google_shot_source_record_id(exp_dir, shot_index)
    rule_id = "google.b8_padding_bits_zero"
    return {
        "issue_id": stable_record_hash([rule_id, source_record_id, member]),
        "run_id": run_id,
        "source_record_id": source_record_id,
        "rule_id": rule_id,
        "severity": Severity.ERROR.value,
        "observed_value": f"{exp_dir}/{member} shot_index={shot_index}",
        "action": "excluded_from_silver",
        "reason": f"unused padding bits in a {member} record must be zero",
    }


def build_google_shot_columns(
    zf: zipfile.ZipFile, exp_dir: str, props: dict, run_id: str
) -> tuple[dict[str, list], list[dict]]:
    """Validate every companion file of one experiment; return (column lists, issue rows).

    measurement_bits, sweep_bits and detector_bits are the packed b8 rows
    unchanged; detector_event_count is the number of set bits in detector_bits.
    """
    shots = props["shots"]
    packed = {}
    bad_shots = {}
    for column, filename, key in GOOGLE_B8_FILES:
        path = f"{exp_dir}/{filename}"
        data = _read_member(zf, path)
        bits = props[key]
        record_bytes = check_b8_size(data, bits, shots, path)
        packed[column] = (data, record_bytes)
        for index in dirty_padding_shots(data, bits):
            bad_shots.setdefault(index, filename)

    path = f"{exp_dir}/obs_flips_actual.01"
    actual = read_01_flags(_read_member(zf, path), shots, path)
    predictions = {}
    for name in GOOGLE_DECODERS:
        path = f"{exp_dir}/obs_flips_predicted_by_{name}.01"
        predictions[name] = read_01_flags(_read_member(zf, path), shots, path)

    columns = {field.name: [] for field in GOOGLE_SHOT_SCHEMA}
    issues = []
    for index in range(shots):
        if index in bad_shots:
            issues.append(_padding_issue(run_id, exp_dir, index, bad_shots[index]))
            continue
        row = {
            column: data[index * record_bytes : (index + 1) * record_bytes]
            for column, (data, record_bytes) in packed.items()
        }
        columns["source_record_id"].append(google_shot_source_record_id(exp_dir, index))
        columns["experiment_id"].append(exp_dir)
        columns["shot_index"].append(index)
        columns["measurement_bits"].append(row["measurement_bits"])
        columns["sweep_bits"].append(row["sweep_bits"])
        columns["detector_bits"].append(row["detector_bits"])
        columns["detector_event_count"].append(
            int.from_bytes(row["detector_bits"], "little").bit_count()
        )
        columns["actual_observable_flip"].append(actual[index])
        for name in GOOGLE_DECODERS:
            columns[f"{name}_prediction"].append(predictions[name][index])
    return columns, issues


def _add_trace(
    trace: dict[str, list],
    source_record_id: str,
    bronze_object: str,
    member: str,
    locator: str,
    input_sha256: str,
) -> None:
    trace["source_record_id"].append(source_record_id)
    trace["source_name"].append(GOOGLE_SOURCE_NAME)
    trace["bronze_object"].append(bronze_object)
    trace["archive_member"].append(member)
    trace["record_locator"].append(locator)
    trace["input_sha256"].append(input_sha256)

#loops over the 5th experiments returning the experiment and shot table plus the issueds , trace rows and the count of shots read.
def build_google_table(
    zf: zipfile.ZipFile, *, bronze_object: str, input_sha256: str, run_id: str
) -> GoogleSilver:
    """Build experiment table, shot table, trace rows and issue rows for the whole archive.

    A shot is assembled from 8 Bronze members, so it gets 8 trace rows sharing
    one source_record_id. b8 locators are ``shot_index=<n>``; 01 locators are
    ``line=<n+1>``.
    """
    experiment_rows, shot_tables, issues = [], [], []
    trace: dict[str, list] = {name: [] for name in TRACE_SCHEMA.names}
    shots_read = 0

    dirs = google_experiment_dirs(zf.namelist())
    if not dirs:
        raise GoogleSilverError("no experiment directories (properties.yml) found in the archive")
    for exp_dir in dirs:
        props = read_google_properties(zf, exp_dir)
        check_google_experiment_name(exp_dir, props)
        row = build_google_experiment_row(exp_dir, props)
        experiment_rows.append(row)
        _add_trace(
            trace, row["source_record_id"], bronze_object, f"{exp_dir}/properties.yml",
            "properties.yml", input_sha256,
        )

        columns, exp_issues = build_google_shot_columns(zf, exp_dir, props, run_id)
        shots_read += props["shots"]
        issues += exp_issues
        shot_tables.append(pa.table(columns, schema=GOOGLE_SHOT_SCHEMA))

        b8_members = [f"{exp_dir}/{filename}" for _c, filename, _k in GOOGLE_B8_FILES]
        flag_members = [f"{exp_dir}/obs_flips_actual.01"] + [
            f"{exp_dir}/obs_flips_predicted_by_{name}.01" for name in GOOGLE_DECODERS
        ]
        for source_record_id, index in zip(columns["source_record_id"], columns["shot_index"]):
            shot_locator, line_locator = f"shot_index={index}", f"line={index + 1}"
            for member in b8_members:
                _add_trace(trace, source_record_id, bronze_object, member, shot_locator, input_sha256)
            for member in flag_members:
                _add_trace(trace, source_record_id, bronze_object, member, line_locator, input_sha256)

    return GoogleSilver(
        experiment=pa.Table.from_pylist(experiment_rows, schema=GOOGLE_EXPERIMENT_SCHEMA),
        shot=pa.concat_tables(shot_tables),
        trace=pa.table(trace, schema=TRACE_SCHEMA),
        issues=issues,
        shots_read=shots_read,
    )


def _replace_file(path: Path, data: bytes) -> None:
    """Write via a temp file so a failed write never leaves a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_silver_table(settings: Settings, table: pa.Table, object_name: str) -> None:
    """Write one Silver Parquet table to the lake (MinIO bucket or the local lake folder)."""
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    data = buffer.getvalue()
    if settings.lake_backend == "local":
        _replace_file(settings.local_lake_root / object_name, data)
    else:
        minio_client(settings).put_object(
            settings.s3_bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="application/vnd.apache.parquet",
        )


def merge_results_parquet(
    path: Path, new_rows: pa.Table, own_rows: Callable[[pa.Table], pa.Array]
) -> None:
    """Add one source's rows to a results file shared by all sources.

    Rows the source wrote in an earlier run (``own_rows`` is True) are replaced,
    so a second run leaves no duplicates; other sources' rows are kept.
    """
    merged = new_rows
    if path.exists():
        existing = pq.read_table(path)
        kept = existing.filter(pc.invert(own_rows(existing))).cast(new_rows.schema)
        merged = pa.concat_tables([kept, new_rows])
    buffer = io.BytesIO()
    pq.write_table(merged, buffer)
    _replace_file(path, buffer.getvalue())


def _merge_prepare_data_json(path: Path, source: str, payload: dict) -> None:
    """Set results[prepare_data][source] in a shared JSON file, keeping other sources' entries."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.setdefault("prepare_data", {})[source] = payload
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))


def run(run_id: str) -> StageResult:
    """Build the Silver tables. Only google_qec is implemented so far.

    A failed check raises before anything is written. The syndrome and
    QASMBench tables are built by adding their own steps below the Google one.
    """
    settings = Settings.from_environment()
    result = StageResult(stage="prepare_data", run_id=run_id)

    archive, bronze_object, input_sha256 = load_google_archive(settings)
    silver = build_google_table(
        archive, bronze_object=bronze_object, input_sha256=input_sha256, run_id=run_id
    )

    write_silver_table(settings, silver.experiment, SILVER_EXPERIMENT_PATH)
    write_silver_table(settings, silver.shot, SILVER_SHOT_PATH)
    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    merge_results_parquet(
        RESULTS_PART1 / "source_trace.parquet",
        silver.trace,
        lambda table: pc.equal(table["source_name"], GOOGLE_SOURCE_NAME),
    )
    merge_results_parquet(
        RESULTS_PART1 / "data_issues.parquet",
        pa.Table.from_pylist(silver.issues, schema=DATA_ISSUES_SCHEMA),
        lambda table: pc.starts_with(table["rule_id"], "google."),
    )

    result.input_count = silver.experiment.num_rows + silver.shots_read
    result.output_count = silver.experiment.num_rows + silver.shot.num_rows
    result.issue_count = len(silver.issues)
    result.finish()

    _merge_prepare_data_json(
        RESULTS_PART1 / "row_counts.json",
        GOOGLE_SOURCE_NAME,
        {
            "experiment": {
                "read": silver.experiment.num_rows,
                "accepted": silver.experiment.num_rows,
                "rejected": 0,
            },
            "shot": {
                "read": silver.shots_read,
                "accepted": silver.shot.num_rows,
                "rejected": len(silver.issues),
            },
            "trace_rows": silver.trace.num_rows,
        },
    )
    _merge_prepare_data_json(
        RESULTS_PART1 / "run.json",
        GOOGLE_SOURCE_NAME,
        {
            "run_id": run_id,
            "code_revision": _code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hashes": {GOOGLE_SOURCE_NAME: input_sha256},
            "output_row_counts": {
                SILVER_EXPERIMENT_PATH: silver.experiment.num_rows,
                SILVER_SHOT_PATH: silver.shot.num_rows,
                "results/part1/source_trace.parquet (google_qec rows)": silver.trace.num_rows,
                "results/part1/data_issues.parquet (google_qec rows)": len(silver.issues),
            },
        },
    )
    return result
