"""Parse, check, and connect the supplied QEC data.

Apply documented checks, normalize nested/wide data, connect valid
relationships, and write prepared Parquet tables with chosen column names and
types. Write invalid records and the reason for exclusion to a machine-readable
data-issues output. The project README defines where generated files live.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from qiskit import qasm2
from qiskit.circuit import Gate, IfElseOp, QuantumCircuit
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.qasm2.exceptions import QASM2ParseError

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import bronze_inventory, minio_client
from quantum_lake_student.formats import b8_record_bytes, parse_01_records
from quantum_lake_student.models import (
    RESULTS_PART1,
    QualityFinding,
    Severity,
    StageResult,
    code_revision,
    merge_json_file,
    stable_record_hash,
)

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

#this function will later be used for the data issues parquet
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
            bin(int.from_bytes(row["detector_bits"], "little")).count("1")
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


def _read_bronze_object(settings: Settings, object_name: str) -> bytes:
    if settings.lake_backend == "local":
        for root_name in ("bronze", "raw"):
            candidate = settings.local_lake_root / object_name.replace("bronze/", f"{root_name}/", 1)
            if candidate.exists():
                return candidate.read_bytes()
        raise FileNotFoundError(f"Bronze object not found under {settings.local_lake_root}: {object_name}")
    client = minio_client(settings)
    response = client.get_object(settings.s3_bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _write_lake_object(settings: Settings, object_name: str, payload: bytes) -> None:
    if settings.lake_backend == "local":
        path = settings.local_lake_root / object_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return
    client = minio_client(settings)
    client.put_object(
        settings.s3_bucket,
        object_name,
        data=BytesIO(payload),
        length=len(payload),
        content_type="application/octet-stream",
    )


def _write_lake_parquet(
    settings: Settings, object_name: str, rows: list[dict[str, Any]], schema: pa.Schema
) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    buffer = BytesIO()
    pq.write_table(table, buffer)
    _write_lake_object(settings, object_name, buffer.getvalue())


# ---------------------------------------------------------------------------
# QASMBench Silver: circuit.parquet, stabilizer_check.parquet, conditional_correction.parquet
# ---------------------------------------------------------------------------

SOURCE_NAME = "qasmbench"
BRONZE_OBJECT = "bronze/source=qasmbench/qasmbench-qec.zip"

_STANDARD_GATE_NAMES = frozenset(get_standard_gate_name_mapping())
_NON_GATE_OP_NAMES = frozenset({"barrier", "measure", "reset", "if_else", "delay"})
_MAX_GATE_EXPANSION_PASSES = 8

_CIRCUIT_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("benchmark_name", pa.string()),
        ("variant", pa.string()),
        ("register_declarations", pa.string()),
        ("qubit_count", pa.int32()),
        ("measurement_count", pa.int32()),
        ("two_qubit_gate_count", pa.int32()),
    ]
)

_STABILIZER_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("check_id", pa.string()),
        ("ancilla_qubit", pa.string()),
        ("data_qubits", pa.list_(pa.string())),
        ("syndrome_bit", pa.string()),
    ]
)

_CORRECTION_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("condition_register", pa.string()),
        ("condition_value", pa.int64()),
        ("gate", pa.string()),
        ("target_qubit", pa.string()),
    ]
)


def prepare_qasmbench_silver(settings: Settings, run_id: str) -> StageResult:
    """Parse the QASMBench Bronze archive into the three QASMBench Silver tables.

    One .qasm member is one circuit variant. A member that cannot be decoded or
    parsed is recorded in data_issues and contributes no rows to any table (a
    circuit is published whole or not at all). Also records this source's
    per-table counts in row_counts.json and its run facts in run.json.
    """
    result = StageResult(stage="prepare_data:qasmbench", run_id=run_id)

    archive_bytes = _read_bronze_object(settings, BRONZE_OBJECT)
    input_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    circuit_rows: list[dict[str, Any]] = []
    stabilizer_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    issues: list[QualityFinding] = []

    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".qasm"))
        for member in members:
            result.input_count += 1
            # every member read gets a trace row, so a rejected circuit's issue
            # row can be traced back to Bronze just like an accepted one
            trace_rows.append(
                _trace_row(_circuit_source_record_id(member), member, member, input_sha256)
            )
            try:
                text = archive.read(member).decode("utf-8")
                parsed = _process_member(member=member, text=text, input_sha256=input_sha256)
            except (ValueError, KeyError, QASM2ParseError) as error:
                # UnicodeDecodeError is a ValueError, so undecodable files land here too
                issues.append(
                    QualityFinding(
                        rule_id="qasmbench.parse_failed",
                        severity=Severity.ERROR,
                        source_system=SOURCE_NAME,
                        source_record_locator=member,
                        message=str(error),
                        observed_value=member,
                    )
                )
                continue
            circuit_rows.append(parsed.circuit)
            stabilizer_rows.extend(parsed.stabilizer_checks)
            correction_rows.extend(parsed.corrections)
            trace_rows.extend(parsed.trace)

    result.output_count = len(circuit_rows) + len(stabilizer_rows) + len(correction_rows)
    result.issue_count = len(issues)

    _write_lake_parquet(settings, "silver/qasmbench/circuit.parquet", circuit_rows, _CIRCUIT_SCHEMA)
    _write_lake_parquet(
        settings, "silver/qasmbench/stabilizer_check.parquet", stabilizer_rows, _STABILIZER_SCHEMA
    )
    _write_lake_parquet(
        settings,
        "silver/qasmbench/conditional_correction.parquet",
        correction_rows,
        _CORRECTION_SCHEMA,
    )

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    merge_results_parquet(
        RESULTS_PART1 / "source_trace.parquet",
        pa.Table.from_pylist(trace_rows, schema=TRACE_SCHEMA),
        lambda table: pc.equal(table["source_name"], SOURCE_NAME),
    )
    merge_results_parquet(
        RESULTS_PART1 / "data_issues.parquet",
        pa.Table.from_pylist(
            [_issue_row(run_id, finding) for finding in issues], schema=DATA_ISSUES_SCHEMA
        ),
        lambda table: pc.starts_with(table["rule_id"], "qasmbench."),
    )

    result.finish()

    # stabilizer checks and corrections are derived from accepted circuits only, so a rejected member removes its rows from all three tables at once
    rejected_members = len(issues)
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "prepare_data",
        SOURCE_NAME,
        payload={
            "circuit": {
                "read": result.input_count,
                "accepted": len(circuit_rows),
                "rejected": rejected_members,
            },
            "stabilizer_check": {"accepted": len(stabilizer_rows)},
            "conditional_correction": {"accepted": len(correction_rows)},
            "trace_rows": len(trace_rows),
        },
    )
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "prepare_data",
        SOURCE_NAME,
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hashes": {SOURCE_NAME: input_sha256},
            "output_row_counts": {
                "silver/qasmbench/circuit.parquet": len(circuit_rows),
                "silver/qasmbench/stabilizer_check.parquet": len(stabilizer_rows),
                "silver/qasmbench/conditional_correction.parquet": len(correction_rows),
                "results/part1/source_trace.parquet (qasmbench rows)": len(trace_rows),
                "results/part1/data_issues.parquet (qasmbench rows)": len(issues),
            },
        },
    )
    return result


@dataclass
class _ParsedCircuit:
    circuit: dict[str, Any]
    stabilizer_checks: list[dict[str, Any]]
    corrections: list[dict[str, Any]]
    trace: list[dict[str, Any]]


def _circuit_source_record_id(member: str) -> str:
    return f"qasmbench:{member}"


def _process_member(*, member: str, text: str, input_sha256: str) -> _ParsedCircuit:
    parts = member.split("/")
    benchmark_name = parts[-2]
    variant = "transpiled" if parts[-1].endswith("_transpiled.qasm") else "source"

    circuit = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    expanded = _expand_custom_gates(circuit)

    circuit_source_record_id = _circuit_source_record_id(member)
    circuit_id = f"{benchmark_name}:{variant}"
    measurement_count, two_qubit_gate_count = _circuit_counts(expanded)

    parsed = _ParsedCircuit(
        circuit={
            "source_record_id": circuit_source_record_id,
            "circuit_id": circuit_id,
            "benchmark_name": benchmark_name,
            "variant": variant,
            "register_declarations": _register_declarations(circuit),
            "qubit_count": sum(register.size for register in circuit.qregs),
            "measurement_count": measurement_count,
            "two_qubit_gate_count": two_qubit_gate_count,
        },
        stabilizer_checks=[],
        corrections=[],
        trace=[],
    )

    for ancilla_qubit, data_qubits, syndrome_bit, occurrence in _find_stabilizer_checks(expanded):
        check_id = f"{ancilla_qubit}#{occurrence}"
        source_record_id = f"{circuit_source_record_id}::stabilizer::{check_id}"
        parsed.stabilizer_checks.append(
            {
                "source_record_id": source_record_id,
                "circuit_id": circuit_id,
                "check_id": check_id,
                "ancilla_qubit": ancilla_qubit,
                "data_qubits": data_qubits,
                "syndrome_bit": syndrome_bit,
            }
        )
        parsed.trace.append(
            _trace_row(source_record_id, member, f"{member}#check:{check_id}", input_sha256)
        )

    for index, (condition_register, condition_value, gate, target_qubit) in enumerate(
        _find_conditional_corrections(expanded)
    ):
        source_record_id = f"{circuit_source_record_id}::correction::{index}"
        parsed.corrections.append(
            {
                "source_record_id": source_record_id,
                "circuit_id": circuit_id,
                "condition_register": condition_register,
                "condition_value": condition_value,
                "gate": gate,
                "target_qubit": target_qubit,
            }
        )
        parsed.trace.append(
            _trace_row(
                source_record_id,
                member,
                f"{member}#if:{condition_register}=={condition_value}#{index}",
                input_sha256,
            )
        )

    return parsed


def _expand_custom_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    expanded = circuit
    for _ in range(_MAX_GATE_EXPANSION_PASSES):
        custom_names = sorted(
            {
                instruction.operation.name
                for instruction in expanded.data
                if instruction.operation.name not in _STANDARD_GATE_NAMES
                and instruction.operation.name not in _NON_GATE_OP_NAMES
            }
        )
        if not custom_names:
            return expanded
        expanded = expanded.decompose(gates_to_decompose=custom_names)
    raise ValueError("QASM custom gate definitions did not resolve to standard gates")


def _bit_label(circuit: QuantumCircuit, bit: object) -> str:
    location = circuit.find_bit(bit)
    register, index = location.registers[0]
    return f"{register.name}[{index}]"


def _circuit_counts(expanded: QuantumCircuit) -> tuple[int, int]:
    measurement_count = sum(1 for instruction in expanded.data if instruction.operation.name == "measure")
    two_qubit_gate_count = sum(
        1
        for instruction in expanded.data
        if isinstance(instruction.operation, Gate) and instruction.operation.num_qubits == 2
    )
    return measurement_count, two_qubit_gate_count


def _register_declarations(circuit: QuantumCircuit) -> str:
    return json.dumps(
        {
            "qregs": [{"name": register.name, "size": register.size} for register in circuit.qregs],
            "cregs": [{"name": register.name, "size": register.size} for register in circuit.cregs],
        },
        separators=(",", ":"),
    )


def _find_stabilizer_checks(
    expanded: QuantumCircuit,
) -> list[tuple[str, list[str], str, int]]:
    ineligible: set[str] = set()
    for instruction in expanded.data:
        operation = instruction.operation
        if not isinstance(operation, Gate):
            continue
        if operation.num_qubits == 2 and operation.name == "cx":
            ineligible.add(_bit_label(expanded, instruction.qubits[0]))
        elif operation.num_qubits == 1:
            ineligible.add(_bit_label(expanded, instruction.qubits[0]))
        else:
            for qubit in instruction.qubits:
                ineligible.add(_bit_label(expanded, qubit))

    checks: list[tuple[str, list[str], str, int]] = []
    accumulated: dict[str, list[str]] = {}
    occurrence: dict[str, int] = {}
    for instruction in expanded.data:
        operation = instruction.operation
        if isinstance(operation, Gate) and operation.name == "cx" and operation.num_qubits == 2:
            control_label = _bit_label(expanded, instruction.qubits[0])
            target_label = _bit_label(expanded, instruction.qubits[1])
            if target_label in ineligible:
                continue
            accumulated.setdefault(target_label, []).append(control_label)
        elif operation.name == "measure":
            qubit_label = _bit_label(expanded, instruction.qubits[0])
            data_qubits = accumulated.pop(qubit_label, [])
            if data_qubits:
                syndrome_bit = _bit_label(expanded, instruction.clbits[0])
                next_occurrence = occurrence.get(qubit_label, -1) + 1
                occurrence[qubit_label] = next_occurrence
                checks.append((qubit_label, data_qubits, syndrome_bit, next_occurrence))
    return checks


def _find_conditional_corrections(
    expanded: QuantumCircuit,
) -> list[tuple[str, int, str, str]]:
    corrections: list[tuple[str, int, str, str]] = []
    for instruction in expanded.data:
        operation = instruction.operation
        if not isinstance(operation, IfElseOp) or operation.condition is None:
            continue
        condition_target, condition_value = operation.condition
        condition_register = getattr(condition_target, "name", None)
        if condition_register is None:
            continue
        block = operation.blocks[0]
        qubit_map = dict(zip(block.qubits, instruction.qubits))
        for body_instruction in block.data:
            body_operation = body_instruction.operation
            if not isinstance(body_operation, Gate) or body_operation.num_qubits != 1:
                continue
            outer_qubit = qubit_map[body_instruction.qubits[0]]
            corrections.append(
                (
                    condition_register,
                    int(condition_value),
                    body_operation.name,
                    _bit_label(expanded, outer_qubit),
                )
            )
    return corrections


def _trace_row(source_record_id: str, archive_member: str, record_locator: str, input_sha256: str) -> dict[str, Any]:
    return {
        "source_record_id": source_record_id,
        "source_name": SOURCE_NAME,
        "bronze_object": BRONZE_OBJECT,
        "archive_member": archive_member,
        "record_locator": record_locator,
        "input_sha256": input_sha256,
    }


def _issue_row(run_id: str, finding: QualityFinding) -> dict[str, Any]:
    return {
        # content hash, not run_id: a rerun on unchanged input gives the same id
        "issue_id": stable_record_hash(
            [finding.rule_id, finding.source_record_locator, finding.message]
        )[:32],
        "run_id": run_id,
        "source_record_id": _circuit_source_record_id(finding.source_record_locator),
        "rule_id": finding.rule_id,
        "severity": finding.severity.value,
        "observed_value": finding.observed_value,
        "action": "excluded_from_silver",
        "reason": finding.message,
    }


# ---------------------------------------------------------------------------
# qec_syndromes Silver: syndrome_observation.parquet
# ---------------------------------------------------------------------------

SYNDROMES_SOURCE_NAME = "qec_syndromes"
SYNDROMES_BRONZE_OBJECT = "bronze/source=qec_syndromes/syndromes_dataset.zip"
SYNDROMES_EXPECTED_HEADER = "labels,syndromes,quantity"
SYNDROMES_FILENAME_RE = re.compile(r"^d-(\d+)_pfr-([\d.]+)_nb-(\w+)\.csv$")

SYNDROMES_SILVER_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("experiment_id", pa.string()),
    ("physical_fault_rate", pa.float64()),
    ("syndrome_bits", pa.binary()),  # 4 rounds x 4 checks, one byte per bit (always 16 bytes)
    ("round_count", pa.int32()),
    ("check_count", pa.int32()),
    ("logical_error_label", pa.bool_()),
    ("quantity", pa.int64()),
    # extra column (allowed by silver-tables.md): code distance from the
    # filename's d-<distance> part, so Gold can label experiments without
    # re-reading Bronze filenames
    ("distance", pa.int32()),
])

SILVER_SYNDROMES_PATH = "silver/qec_syndromes/syndrome_observation.parquet"


@dataclass(frozen=True)
class DataIssue:
    # why a record was rejected
    source_record_id: str | None
    rule_id: str
    severity: str
    observed_value: str
    action: str
    reason: str


@dataclass(frozen=True)
class TraceRow:
    # trace the row from the original file to its new Silver position
    source_record_id: str
    source_name: str
    bronze_object: str
    archive_member: str
    record_locator: str
    input_sha256: str


def _syndromes_experiment_id(member: str) -> str:
    # labels rows with which file they came from
    return f"qec_syndromes:{member}"


def _syndromes_record_id(member: str, row_index: int) -> str:
    # Hash of file+position, not random: reruns produce the same id for the
    # same row, so a second run doesn't create duplicates
    return hashlib.sha256(f"{member}:{row_index}".encode()).hexdigest()[:32]


def _syndromes_parse_bits(raw: str) -> list[list[int]] | None:
    # Parse the syndromes column text and validate it is a 4x4 grid of 0/1 values.
    # Returns None if anything doesn't look right.
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, tuple) or len(parsed) != 4:
        return None
    rounds: list[list[int]] = []
    for round_tuple in parsed:
        if not isinstance(round_tuple, tuple) or len(round_tuple) != 4:
            return None
        if not all(bit in (0, 1) for bit in round_tuple):
            return None
        rounds.append(list(round_tuple))
    return rounds


def _syndromes_pack_bits(rounds: list[list[int]]) -> bytes:
    # Flatten the 4x4 nested structure into 16 ordered bytes.
    return bytes(bit for round_bits in rounds for bit in round_bits)


def build_syndromes(bronze_bytes: bytes, input_sha256: str) -> tuple[
    pa.Table, list[DataIssue], list[TraceRow]
]:
    # opens the zip and finds every .csv file
    records: list[dict] = []
    issues: list[DataIssue] = []
    trace_rows: list[TraceRow] = []

    with zipfile.ZipFile(io.BytesIO(bronze_bytes)) as archive:
        csv_members = sorted(n for n in archive.namelist() if n.endswith(".csv"))
        # checks if the filename and header look right
        for member in csv_members:
            match = SYNDROMES_FILENAME_RE.match(member)
            if not match:
                issues.append(DataIssue(
                    source_record_id=None,
                    rule_id="syndromes.filename_format",
                    severity="error",
                    observed_value=member,
                    action="file_skipped",
                    reason="filename does not match d-<distance>_pfr-<rate>_nb-<n>.csv",
                ))
                continue
            distance = int(match.group(1))
            fault_rate = float(match.group(2))
            experiment_id = _syndromes_experiment_id(member)

            with archive.open(member) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8")
                try:
                    header = next(text).strip()
                except StopIteration:
                    issues.append(DataIssue(
                        source_record_id=None,
                        rule_id="syndromes.empty_file",
                        severity="error",
                        observed_value=member,
                        action="file_skipped",
                        reason="file is empty (no header row)",
                    ))
                    continue
                if header != SYNDROMES_EXPECTED_HEADER:
                    issues.append(DataIssue(
                        source_record_id=None,
                        rule_id="syndromes.header",
                        severity="error",
                        observed_value=header,
                        action="file_skipped",
                        reason=f"{member}: expected header '{SYNDROMES_EXPECTED_HEADER}'",
                    ))
                    continue

                # reads it and for each line it splits out the 3 fields
                accepted_quantity_total = 0
                for row_index, line in enumerate(text, start=1):
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    # records a trace entry for the row
                    record_id = _syndromes_record_id(member, row_index)
                    trace_rows.append(TraceRow(
                        source_record_id=record_id,
                        source_name=SYNDROMES_SOURCE_NAME,
                        bronze_object=SYNDROMES_BRONZE_OBJECT,
                        archive_member=member,
                        record_locator=str(row_index),
                        input_sha256=input_sha256,
                    ))

                    # quantity is the last comma field; syndromes is quoted and
                    # may itself contain commas, so split from the right
                    try:
                        label_str, rest = line.split(",", 1)
                        syndromes_str, quantity_str = rest.rsplit(",", 1)
                    except ValueError:
                        issues.append(DataIssue(
                            source_record_id=record_id,
                            rule_id="syndromes.row_format",
                            severity="error",
                            observed_value=line,
                            action="row_excluded",
                            reason="row does not have the expected label,syndromes,quantity format",
                        ))
                        continue
                    syndromes_str = syndromes_str.strip('"')

                    # runs the validity checks
                    rounds = _syndromes_parse_bits(syndromes_str)
                    try:
                        quantity = int(quantity_str)
                    except ValueError:
                        quantity = None
                    label = label_str.strip()

                    if rounds is None:
                        issues.append(DataIssue(
                            source_record_id=record_id,
                            rule_id="syndromes.shape",
                            severity="error",
                            observed_value=syndromes_str,
                            action="row_excluded",
                            reason="syndromes value is not a valid 4x4 binary tuple",
                        ))
                        continue
                    if quantity is None or quantity <= 0:
                        issues.append(DataIssue(
                            source_record_id=record_id,
                            rule_id="syndromes.quantity_positive",
                            severity="error",
                            observed_value=quantity_str,
                            action="row_excluded",
                            reason="quantity must be a positive integer",
                        ))
                        continue
                    if label not in ("0", "1"):
                        issues.append(DataIssue(
                            source_record_id=record_id,
                            rule_id="syndromes.label_domain",
                            severity="error",
                            observed_value=label,
                            action="row_excluded",
                            reason="labels must be 0 or 1",
                        ))
                        continue

                    accepted_quantity_total += quantity
                    records.append({
                        "source_record_id": record_id,
                        "experiment_id": experiment_id,
                        "physical_fault_rate": fault_rate,
                        "syndrome_bits": _syndromes_pack_bits(rounds),
                        "round_count": 4,
                        "check_count": 4,
                        "logical_error_label": bool(int(label)),
                        "quantity": quantity,
                        "distance": distance,
                    })

                # checks that the total quantity of everything kept
                # roughly matches what the filename promised, so as to
                # not silently lose or duplicate anything
                nb_hint = match.group(3)
                if nb_hint.endswith("M"):
                    expected_total = float(nb_hint[:-1]) * 1_000_000
                    if abs(accepted_quantity_total - expected_total) > 0.02 * expected_total:
                        issues.append(DataIssue(
                            source_record_id=None,
                            rule_id="syndromes.weighted_total_reconciliation",
                            severity="warning",
                            observed_value=str(accepted_quantity_total),
                            action="file_flagged",
                            reason=(
                                f"{member}: accepted weighted total "
                                f"{accepted_quantity_total} vs filename hint "
                                f"{expected_total}"
                            ),
                        ))

    # builds one clean table from everything that passed
    table = pa.Table.from_pylist(records, schema=SYNDROMES_SILVER_SCHEMA)
    return table, issues, trace_rows


def _issue_id(issue: DataIssue) -> str:
    # hash of the issue's own content, not its position in the list: a rerun
    # that finds the same problem (in any order) gets the same issue_id
    key = f"{issue.rule_id}:{issue.source_record_id or ''}:{issue.observed_value}:{issue.reason}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _syndromes_issue_rows(run_id: str, issues: list[DataIssue]) -> list[dict]:
    return [
        {
            "issue_id": _issue_id(issue),
            "run_id": run_id,
            "source_record_id": issue.source_record_id,
            "rule_id": issue.rule_id,
            "severity": issue.severity,
            "observed_value": issue.observed_value,
            "action": issue.action,
            "reason": issue.reason,
        }
        for issue in issues
    ]


def run(run_id: str) -> StageResult:
    """Build the Silver tables for google_qec, qec_syndromes, and qasmbench.

    A google_qec check that makes an experiment unusable raises before
    anything is written for that source. A qec_syndromes row or file that
    fails validation is recorded as a data issue and excluded, never raised;
    a second run of either source replaces only that source's rows in the
    shared results files, leaving the other source's rows untouched.
    """
    settings = Settings.from_environment()
    result = StageResult(stage="prepare_data", run_id=run_id)

    # ---- google_qec ----
    archive, bronze_object, google_input_sha256 = load_google_archive(settings)
    google_silver = build_google_table(
        archive, bronze_object=bronze_object, input_sha256=google_input_sha256, run_id=run_id
    )

    write_silver_table(settings, google_silver.experiment, SILVER_EXPERIMENT_PATH)
    write_silver_table(settings, google_silver.shot, SILVER_SHOT_PATH)
    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    merge_results_parquet(
        RESULTS_PART1 / "source_trace.parquet",
        google_silver.trace,
        lambda table: pc.equal(table["source_name"], GOOGLE_SOURCE_NAME),
    )
    merge_results_parquet(
        RESULTS_PART1 / "data_issues.parquet",
        pa.Table.from_pylist(google_silver.issues, schema=DATA_ISSUES_SCHEMA),
        lambda table: pc.starts_with(table["rule_id"], "google."),
    )

    result.input_count += google_silver.experiment.num_rows + google_silver.shots_read
    result.output_count += google_silver.experiment.num_rows + google_silver.shot.num_rows
    result.issue_count += len(google_silver.issues)

    # ---- qec_syndromes ----
    syndromes_bytes = _read_bronze_object(settings, SYNDROMES_BRONZE_OBJECT)
    syndromes_input_sha256 = hashlib.sha256(syndromes_bytes).hexdigest()
    syndromes_table, syndromes_issues, syndromes_trace = build_syndromes(
        syndromes_bytes, input_sha256=syndromes_input_sha256
    )
    # only row-level issues count as rejected rows; file-level issues (skipped
    # file, weighted-total warning) are not rows, so read == accepted + rejected
    syndromes_rows_rejected = sum(
        1 for issue in syndromes_issues if issue.action == "row_excluded"
    )
    syndromes_rows_read = syndromes_table.num_rows + syndromes_rows_rejected

    write_silver_table(settings, syndromes_table, SILVER_SYNDROMES_PATH)
    merge_results_parquet(
        RESULTS_PART1 / "source_trace.parquet",
        pa.Table.from_pylist([row.__dict__ for row in syndromes_trace], schema=TRACE_SCHEMA),
        lambda table: pc.equal(table["source_name"], SYNDROMES_SOURCE_NAME),
    )
    merge_results_parquet(
        RESULTS_PART1 / "data_issues.parquet",
        pa.Table.from_pylist(
            _syndromes_issue_rows(run_id, syndromes_issues), schema=DATA_ISSUES_SCHEMA
        ),
        lambda table: pc.starts_with(table["rule_id"], "syndromes."),
    )

    result.input_count += syndromes_rows_read
    result.output_count += syndromes_table.num_rows
    result.issue_count += len(syndromes_issues)

    # ---- qasmbench ----
    qasmbench_result = prepare_qasmbench_silver(settings, run_id)
    result.input_count += qasmbench_result.input_count
    result.output_count += qasmbench_result.output_count
    result.issue_count += qasmbench_result.issue_count

    result.finish()

    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "prepare_data",
        GOOGLE_SOURCE_NAME,
        payload={
            "experiment": {
                "read": google_silver.experiment.num_rows,
                "accepted": google_silver.experiment.num_rows,
                "rejected": 0,
            },
            "shot": {
                "read": google_silver.shots_read,
                "accepted": google_silver.shot.num_rows,
                "rejected": len(google_silver.issues),
            },
            "trace_rows": google_silver.trace.num_rows,
        },
    )
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "prepare_data",
        GOOGLE_SOURCE_NAME,
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hashes": {GOOGLE_SOURCE_NAME: google_input_sha256},
            "output_row_counts": {
                SILVER_EXPERIMENT_PATH: google_silver.experiment.num_rows,
                SILVER_SHOT_PATH: google_silver.shot.num_rows,
                "results/part1/source_trace.parquet (google_qec rows)": google_silver.trace.num_rows,
                "results/part1/data_issues.parquet (google_qec rows)": len(google_silver.issues),
            },
        },
    )

    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "prepare_data",
        SYNDROMES_SOURCE_NAME,
        payload={
            "read": syndromes_rows_read,
            "accepted": syndromes_table.num_rows,
            "rejected": syndromes_rows_rejected,
        },
    )
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "prepare_data",
        SYNDROMES_SOURCE_NAME,
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hashes": {SYNDROMES_SOURCE_NAME: syndromes_input_sha256},
            "output_row_counts": {
                SILVER_SYNDROMES_PATH: syndromes_table.num_rows,
                "results/part1/source_trace.parquet (qec_syndromes rows)": len(syndromes_trace),
                "results/part1/data_issues.parquet (qec_syndromes rows)": len(syndromes_issues),
            },
        },
    )
    return result
