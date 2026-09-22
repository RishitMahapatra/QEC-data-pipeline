import shutil
from pathlib import Path

import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.stages.prepare_data import prepare_qasmbench_silver

_ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "student-bundle"
    / "core"
    / "raw"
    / "source=qasmbench"
    / "qasmbench-qec.zip"
)


def _settings_with_seeded_archive(tmp_path: Path, monkeypatch) -> Settings:
    destination = tmp_path / "raw" / "source=qasmbench" / "qasmbench-qec.zip"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(_ARCHIVE, destination)
    monkeypatch.setenv("LAKE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LAKE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return Settings.from_environment()


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
