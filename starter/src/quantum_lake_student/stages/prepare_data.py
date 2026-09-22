"""Parse, check, and connect the supplied QEC data.

Apply documented checks, normalize nested/wide data, connect valid
relationships, and write prepared Parquet tables with chosen column names and
types. Write invalid records and the reason for exclusion to a machine-readable
data-issues output. The project README defines where generated files live.

Only the QASMBench Silver tables (circuit, stabilizer_check,
conditional_correction) are implemented here. The syndrome and Google Silver
tables are a teammate's responsibility and are intentionally left as a TODO so
this module can be reviewed and merged independently.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from qiskit import qasm2
from qiskit.circuit import Gate, IfElseOp, QuantumCircuit
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.qasm2.exceptions import QASM2ParseError

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.models import QualityFinding, Severity, StageResult

SOURCE_NAME = "qasmbench"
BRONZE_OBJECT = "bronze/source=qasmbench/qasmbench-qec.zip"
RESULTS_DIR = Path("results/part1")

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

_TRACE_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("source_name", pa.string()),
        ("bronze_object", pa.string()),
        ("archive_member", pa.string()),
        ("record_locator", pa.string()),
        ("input_sha256", pa.string()),
    ]
)

_ISSUE_SCHEMA = pa.schema(
    [
        ("issue_id", pa.string()),
        ("run_id", pa.string()),
        ("source_name", pa.string()),
        ("source_record_id", pa.string()),
        ("rule_id", pa.string()),
        ("severity", pa.string()),
        ("observed_value", pa.string()),
        ("action", pa.string()),
        ("reason", pa.string()),
    ]
)


def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    # TODO(teammate): add syndrome and Google Silver preparation here and
    # merge their StageResult counts with this one.
    return prepare_qasmbench_silver(settings, run_id)


def prepare_qasmbench_silver(settings: Settings, run_id: str) -> StageResult:
    """Parse the QASMBench Bronze archive into the three QASMBench Silver tables."""
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
            text = archive.read(member).decode("utf-8")
            try:
                _process_member(
                    member=member,
                    text=text,
                    input_sha256=input_sha256,
                    circuit_rows=circuit_rows,
                    stabilizer_rows=stabilizer_rows,
                    correction_rows=correction_rows,
                    trace_rows=trace_rows,
                )
            except (ValueError, KeyError, QASM2ParseError) as error:
                issues.append(
                    QualityFinding(
                        rule_id="qasmbench.parse_failed",
                        severity=Severity.ERROR,
                        source_system=SOURCE_NAME,
                        source_record_locator=member,
                        message=str(error),
                    )
                )

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

    _merge_source_rows_into(
        RESULTS_DIR / "source_trace.parquet", _TRACE_SCHEMA, trace_rows, source_name_column="source_name"
    )
    _merge_source_rows_into(
        RESULTS_DIR / "data_issues.parquet",
        _ISSUE_SCHEMA,
        [_issue_row(run_id, finding) for finding in issues],
        source_name_column="source_name",
    )

    result.finish()
    return result


def _process_member(
    *,
    member: str,
    text: str,
    input_sha256: str,
    circuit_rows: list[dict[str, Any]],
    stabilizer_rows: list[dict[str, Any]],
    correction_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> None:
    parts = member.split("/")
    benchmark_name = parts[-2]
    variant = "transpiled" if parts[-1].endswith("_transpiled.qasm") else "source"

    # Qiskit's bundled "legacy" instruction set covers every gate the archive's
    # own qelib1.inc defines, plus IBM basis gates (e.g. sx) that appear in the
    # transpiled variants but are not textually defined in that file. This is
    # a parsing convenience only: the circuits are never executed or simulated.
    circuit = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    expanded = _expand_custom_gates(circuit)

    circuit_source_record_id = f"qasmbench:{member}"
    circuit_id = f"{benchmark_name}:{variant}"
    measurement_count, two_qubit_gate_count = _circuit_counts(expanded)

    circuit_rows.append(
        {
            "source_record_id": circuit_source_record_id,
            "circuit_id": circuit_id,
            "benchmark_name": benchmark_name,
            "variant": variant,
            "register_declarations": _register_declarations(circuit),
            "qubit_count": sum(register.size for register in circuit.qregs),
            "measurement_count": measurement_count,
            "two_qubit_gate_count": two_qubit_gate_count,
        }
    )
    trace_rows.append(_trace_row(circuit_source_record_id, member, member, input_sha256))

    for ancilla_qubit, data_qubits, syndrome_bit, occurrence in _find_stabilizer_checks(expanded):
        check_id = f"{ancilla_qubit}#{occurrence}"
        source_record_id = f"{circuit_source_record_id}::stabilizer::{check_id}"
        stabilizer_rows.append(
            {
                "source_record_id": source_record_id,
                "circuit_id": circuit_id,
                "check_id": check_id,
                "ancilla_qubit": ancilla_qubit,
                "data_qubits": data_qubits,
                "syndrome_bit": syndrome_bit,
            }
        )
        trace_rows.append(
            _trace_row(source_record_id, member, f"{member}#check:{check_id}", input_sha256)
        )

    for index, (condition_register, condition_value, gate, target_qubit) in enumerate(
        _find_conditional_corrections(expanded)
    ):
        source_record_id = f"{circuit_source_record_id}::correction::{index}"
        correction_rows.append(
            {
                "source_record_id": source_record_id,
                "circuit_id": circuit_id,
                "condition_register": condition_register,
                "condition_value": condition_value,
                "gate": gate,
                "target_qubit": target_qubit,
            }
        )
        trace_rows.append(
            _trace_row(
                source_record_id,
                member,
                f"{member}#if:{condition_register}=={condition_value}#{index}",
                input_sha256,
            )
        )


def _expand_custom_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """Inline user-defined (non-standard) gate calls into their primitive operations.

    Standard library gates (cx, x, rz, ...) are left untouched so the reported
    operation names match the source QASM. Only genuinely custom gates, such as
    the archive's ``syndrome`` macro, are unrolled.
    """
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
    """Detect parity-check ancillas: qubits used only as CNOT targets, then measured.

    A qubit that is ever a CNOT control, or touched by any other gate, is
    disqualified from the ancilla role -- that behavior marks it as a
    computational qubit rather than a pure parity accumulator. Returns
    (ancilla_qubit, data_qubits, syndrome_bit, occurrence_index) tuples in the
    order the checks close (i.e. are measured).
    """
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
    """Return (condition_register, condition_value, gate, target_qubit) for each if-body gate."""
    corrections: list[tuple[str, int, str, str]] = []
    for instruction in expanded.data:
        operation = instruction.operation
        if not isinstance(operation, IfElseOp) or operation.condition is None:
            continue
        condition_target, condition_value = operation.condition
        condition_register = getattr(condition_target, "name", None)
        if condition_register is None:
            # A single-Clbit condition; not produced by this archive's circuits.
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
        "issue_id": f"{run_id}:{finding.rule_id}:{finding.source_record_locator}",
        "run_id": run_id,
        "source_name": finding.source_system,
        "source_record_id": None,
        "rule_id": finding.rule_id,
        "severity": finding.severity.value,
        "observed_value": finding.observed_value,
        "action": "excluded_from_silver",
        "reason": finding.message,
    }


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


def _merge_source_rows_into(
    path: Path, schema: pa.Schema, new_rows: list[dict[str, Any]], *, source_name_column: str
) -> None:
    """Overwrite this source's rows in a shared results table without disturbing other sources'.

    `source_trace.parquet` and `data_issues.parquet` are populated by every
    source's stage. A second run on unchanged input must reproduce the same
    QASMBench rows, but must not drop rows another source already wrote.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path)
        kept_mask = pc.not_equal(existing.column(source_name_column), SOURCE_NAME)
        existing = existing.filter(kept_mask)
    else:
        existing = pa.Table.from_pylist([], schema=schema)
    new_table = pa.Table.from_pylist(new_rows, schema=schema)
    combined = pa.concat_tables([existing, new_table])
    pq.write_table(combined, path)
