"""Tests for the Silver prepare_data stage: qasmbench.

The first two tests run on the real course archive (skipped if it is not
available). The rest build small in-memory zips, so they always run.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.stages.prepare_data import prepare_qasmbench_silver

_ARCHIVE_RELATIVE = Path("raw") / "source=qasmbench" / "qasmbench-qec.zip"
# The bundle is mounted at /course-data in the workspace container; on the host it
# lives in the repo's datasets/ folder.
_ARCHIVE_CANDIDATES = (
    Path("/course-data") / _ARCHIVE_RELATIVE,
    Path(__file__).resolve().parents[2] / "datasets" / "student-bundle" / "core" / _ARCHIVE_RELATIVE,
)
_ARCHIVE = next((p for p in _ARCHIVE_CANDIDATES if p.exists()), None)

needs_archive = pytest.mark.skipif(
    _ARCHIVE is None, reason="qasmbench-qec.zip not found in the course bundle"
)


def _seed_archive(tmp_path: Path, monkeypatch, archive_bytes: bytes) -> Settings:
    destination = tmp_path / "raw" / "source=qasmbench" / "qasmbench-qec.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(archive_bytes)
    monkeypatch.setenv("LAKE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LAKE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return Settings.from_environment()


def _settings_with_seeded_archive(tmp_path: Path, monkeypatch) -> Settings:
    return _seed_archive(tmp_path, monkeypatch, _ARCHIVE.read_bytes())


def _zip_bytes(members: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _silver(tmp_path: Path, table: str) -> pa.Table:
    return pq.read_table(tmp_path / "silver" / "qasmbench" / f"{table}.parquet")


# qec_sm_n5-shaped circuit: a custom gate whose body holds the parity CNOTs,
# register-wide measurements, and a register-wide conditional correction
_SYNDROME_CIRCUIT = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
creg c[3];
creg syn[2];
gate syndrome d1,d2,d3,a1,a2
{
  cx d1,a1; cx d2,a1;
  cx d2,a2; cx d3,a2;
}
syndrome q[0],q[1],q[2],a[0],a[1];
measure a -> syn;
if(syn==1) x q[0];
if(syn==3) x q;
measure q -> c;
"""


@needs_archive
def test_prepare_qasmbench_silver_writes_the_three_tables(tmp_path: Path, monkeypatch) -> None:
    settings = _settings_with_seeded_archive(tmp_path, monkeypatch)

    result = prepare_qasmbench_silver(settings, run_id="test-run")

    assert result.input_count == 6  # 3 benchmarks x (source, transpiled)
    assert result.issue_count == 0

    circuit = pq.read_table(tmp_path / "silver" / "qasmbench" / "circuit.parquet").to_pylist()
    stabilizer = pq.read_table(
        tmp_path / "silver" / "qasmbench" / "stabilizer_check.parquet"
    ).to_pylist()
    correction = pq.read_table(
        tmp_path / "silver" / "qasmbench" / "conditional_correction.parquet"
    ).to_pylist()

    assert len(circuit) == 6
    assert {row["circuit_id"] for row in circuit} == {
        "error_correctiond3_n5:source",
        "error_correctiond3_n5:transpiled",
        "qec_en_n5:source",
        "qec_en_n5:transpiled",
        "qec_sm_n5:source",
        "qec_sm_n5:transpiled",
    }
    assert all(row["source_record_id"] for row in circuit)

    # Only qec_sm_n5 contains an explicit parity-check/correction structure;
    # the source and transpiled variants must agree on the extracted structure.
    for variant in ("source", "transpiled"):
        checks = sorted(
            (row["ancilla_qubit"], tuple(row["data_qubits"]), row["syndrome_bit"])
            for row in stabilizer
            if row["circuit_id"] == f"qec_sm_n5:{variant}"
        )
        assert checks == [
            ("a[0]", ("q[0]", "q[1]"), "syn[0]"),
            ("a[1]", ("q[1]", "q[2]"), "syn[1]"),
        ]

        corrections = sorted(
            (row["condition_value"], row["gate"], row["target_qubit"])
            for row in correction
            if row["circuit_id"] == f"qec_sm_n5:{variant}"
        )
        assert corrections == [
            (1, "x", "q[0]"),
            (2, "x", "q[2]"),
            (3, "x", "q[1]"),
        ]

    assert not any(row["circuit_id"].startswith("qec_en_n5") for row in stabilizer)
    assert not any(row["circuit_id"].startswith("error_correctiond3_n5") for row in stabilizer)


@needs_archive
def test_prepare_qasmbench_silver_writes_trace_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    settings = _settings_with_seeded_archive(tmp_path, monkeypatch)

    first = prepare_qasmbench_silver(settings, run_id="run-1")
    second = prepare_qasmbench_silver(settings, run_id="run-2")

    trace = pq.read_table(tmp_path / "results" / "part1" / "source_trace.parquet").to_pylist()
    assert all(row["source_name"] == "qasmbench" for row in trace)
    assert first.output_count == second.output_count
    assert {row["source_record_id"] for row in trace} >= {
        row["source_record_id"]
        for row in pq.read_table(tmp_path / "silver" / "qasmbench" / "circuit.parquet").to_pylist()
    }

    issues_path = tmp_path / "results" / "part1" / "data_issues.parquet"
    assert issues_path.exists()
    assert pq.read_table(issues_path).num_rows == 0


def test_schemas_match_silver_contract(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_archive(
        tmp_path, monkeypatch, _zip_bytes({"small/qec_sm_n5/qec_sm_n5.qasm": _SYNDROME_CIRCUIT})
    )
    prepare_qasmbench_silver(settings, run_id="run-1")

    assert _silver(tmp_path, "circuit").schema == pa.schema([
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("benchmark_name", pa.string()),
        ("variant", pa.string()),
        ("register_declarations", pa.string()),
        ("qubit_count", pa.int32()),
        ("measurement_count", pa.int32()),
        ("two_qubit_gate_count", pa.int32()),
    ])
    assert _silver(tmp_path, "stabilizer_check").schema == pa.schema([
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("check_id", pa.string()),
        ("ancilla_qubit", pa.string()),
        ("data_qubits", pa.list_(pa.string())),
        ("syndrome_bit", pa.string()),
    ])
    assert _silver(tmp_path, "conditional_correction").schema == pa.schema([
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("condition_register", pa.string()),
        ("condition_value", pa.int64()),
        ("gate", pa.string()),
        ("target_qubit", pa.string()),
    ])


def test_counts_use_expanded_operations_not_statement_lines(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_archive(
        tmp_path, monkeypatch, _zip_bytes({"small/qec_sm_n5/qec_sm_n5.qasm": _SYNDROME_CIRCUIT})
    )
    prepare_qasmbench_silver(settings, run_id="run-1")

    [circuit] = _silver(tmp_path, "circuit").to_pylist()
    # one `syndrome` statement = 4 executed CNOTs; the gate body itself is a
    # definition and is not counted on its own
    assert circuit["two_qubit_gate_count"] == 4
    # `measure a -> syn` (2) + `measure q -> c` (3), not 2 statement lines
    assert circuit["measurement_count"] == 5
    assert circuit["qubit_count"] == 5
    assert json.loads(circuit["register_declarations"]) == {
        "qregs": [{"name": "q", "size": 3}, {"name": "a", "size": 2}],
        "cregs": [{"name": "c", "size": 3}, {"name": "syn", "size": 2}],
    }


def test_checks_and_register_wide_corrections_are_extracted(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_archive(
        tmp_path, monkeypatch, _zip_bytes({"small/qec_sm_n5/qec_sm_n5.qasm": _SYNDROME_CIRCUIT})
    )
    prepare_qasmbench_silver(settings, run_id="run-1")

    checks = sorted(
        (row["ancilla_qubit"], tuple(row["data_qubits"]), row["syndrome_bit"])
        for row in _silver(tmp_path, "stabilizer_check").to_pylist()
    )
    # a[0] and q[0] are different qubits: register names are kept
    assert checks == [("a[0]", ("q[0]", "q[1]"), "syn[0]"), ("a[1]", ("q[1]", "q[2]"), "syn[1]")]

    corrections = sorted(
        (row["condition_register"], row["condition_value"], row["gate"], row["target_qubit"])
        for row in _silver(tmp_path, "conditional_correction").to_pylist()
    )
    # `if(syn==3) x q;` addresses the whole register -> one row per qubit
    assert corrections == [
        ("syn", 1, "x", "q[0]"),
        ("syn", 3, "x", "q[0]"),
        ("syn", 3, "x", "q[1]"),
        ("syn", 3, "x", "q[2]"),
    ]


def test_unparseable_and_undecodable_members_become_issues(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_archive(
        tmp_path,
        monkeypatch,
        _zip_bytes({
            "small/qec_sm_n5/qec_sm_n5.qasm": _SYNDROME_CIRCUIT,
            "small/broken/broken.qasm": "OPENQASM 2.0;\nqreg q[2;\n",
            "small/binary/binary.qasm": b"\xff\xfe not utf-8",
        }),
    )

    result = prepare_qasmbench_silver(settings, run_id="run-1")

    assert result.input_count == 3
    assert result.issue_count == 2
    # the good circuit is still published; the bad ones add no rows anywhere
    assert _silver(tmp_path, "circuit").column("circuit_id").to_pylist() == ["qec_sm_n5:source"]
    assert set(_silver(tmp_path, "stabilizer_check").column("circuit_id").to_pylist()) == {
        "qec_sm_n5:source"
    }

    issues = pq.read_table(tmp_path / "results" / "part1" / "data_issues.parquet").to_pylist()
    assert sorted(issue["observed_value"] for issue in issues) == [
        "small/binary/binary.qasm",
        "small/broken/broken.qasm",
    ]
    for issue in issues:
        assert issue["rule_id"] == "qasmbench.parse_failed"
        assert issue["action"] == "excluded_from_silver"
        assert issue["reason"]
        assert issue["source_record_id"] == f"qasmbench:{issue['observed_value']}"

    # rejected members are still traceable to Bronze
    trace_ids = set(
        pq.read_table(tmp_path / "results" / "part1" / "source_trace.parquet")
        .column("source_record_id")
        .to_pylist()
    )
    assert {issue["source_record_id"] for issue in issues} <= trace_ids


def test_rerun_keeps_issue_ids_and_counts_reconcile(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_archive(
        tmp_path,
        monkeypatch,
        _zip_bytes({
            "small/qec_sm_n5/qec_sm_n5.qasm": _SYNDROME_CIRCUIT,
            "small/broken/broken.qasm": "OPENQASM 2.0;\nqreg q[2;\n",
        }),
    )
    issues_path = tmp_path / "results" / "part1" / "data_issues.parquet"

    prepare_qasmbench_silver(settings, run_id="run-1")
    first_ids = pq.read_table(issues_path).column("issue_id").to_pylist()
    prepare_qasmbench_silver(settings, run_id="run-2")
    second_ids = pq.read_table(issues_path).column("issue_id").to_pylist()

    # same input -> same issue ids, and no duplicates after a second run
    assert first_ids == second_ids
    assert len(second_ids) == 1

    counts = json.loads((tmp_path / "results" / "part1" / "row_counts.json").read_text())
    qasmbench = counts["prepare_data"]["qasmbench"]
    circuit = qasmbench["circuit"]
    assert circuit == {"read": 2, "accepted": 1, "rejected": 1}
    assert circuit["read"] == circuit["accepted"] + circuit["rejected"]
    assert qasmbench["stabilizer_check"] == {"accepted": 2}
    assert qasmbench["conditional_correction"] == {"accepted": 4}

    run = json.loads((tmp_path / "results" / "part1" / "run.json").read_text())
    qasmbench_run = run["prepare_data"]["qasmbench"]
    assert qasmbench_run["run_id"] == "run-2"
    assert len(qasmbench_run["input_hashes"]["qasmbench"]) == 64
    assert qasmbench_run["output_row_counts"]["silver/qasmbench/circuit.parquet"] == 1
