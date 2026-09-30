"""Tests for the Gold-to-ML export stage (build_ml_tables.py).

The contract-check tests need no database. The other tests load small
in-memory Silver tables for all three sources into a throwaway schema
(test_gold_<random>) with load_postgres.run(), which also creates the ML
views, then run build_ml_tables.run() against that schema. The lake write is
captured in memory and results files go to tmp_path, so nothing real is
touched. The real committed analysis SQL files are copied into tmp_path and
run against the test schema.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import uuid
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.stages import build_ml_tables as bml
from quantum_lake_student.stages import load_postgres as lp
from quantum_lake_student.stages import prepare_data
from quantum_lake_student.stages.prepare_data import SYNDROMES_SILVER_SCHEMA

ANALYSIS_SQL_DIR = Path(__file__).resolve().parent.parent / "results" / "part1" / "analysis"


# ---------------------------------------------------------------------------
# In-memory Silver rows covering all three splits of both ML tables
# ---------------------------------------------------------------------------

PATTERN = bytes([0] * 6 + [1] + [0] * 9)
OTHER_PATTERN = bytes([1] + [0] * 15)


def _syndrome_row(record_id, experiment_id, rate, bits, label, quantity):
    return {
        "source_record_id": record_id, "experiment_id": experiment_id, "physical_fault_rate": rate,
        "syndrome_bits": bits, "round_count": 4, "check_count": 4,
        "logical_error_label": label, "quantity": quantity, "distance": 3,
    }


# train (0.0001, same syndrome with both labels), validation (0.0005), test (0.005)
SYNDROME_ROWS = [
    _syndrome_row("syn-0", "exp-train", 0.0001, PATTERN, False, 100),
    _syndrome_row("syn-1", "exp-train", 0.0001, PATTERN, True, 30),
    _syndrome_row("syn-2", "exp-val", 0.0005, OTHER_PATTERN, False, 50),
    _syndrome_row("syn-3", "exp-test", 0.005, PATTERN, True, 7),
]

GOOGLE_EXPERIMENT = [{
    "source_record_id": "exp-src-1", "experiment_id": "surface_code_bX_d3_r25_center_3_5", "basis": "X",
    "distance": 3, "rounds": 25, "shots": 3, "center_row": 3, "center_col": 5,
    "measurement_count": 10, "detector_count": 200,
}]


def _shot(shot_index, detector_bits, predictions, actual):
    return {
        "source_record_id": f"shot-src-{shot_index}", "experiment_id": "surface_code_bX_d3_r25_center_3_5",
        "shot_index": shot_index, "measurement_bits": b"\x00\x00", "sweep_bits": b"\x00\x00",
        "detector_bits": detector_bits,
        "detector_event_count": int.from_bytes(detector_bits, "little").bit_count(),
        "actual_observable_flip": actual,
        "belief_matching_prediction": predictions[0], "correlated_matching_prediction": predictions[1],
        "pymatching_prediction": predictions[2], "tensor_network_contraction_prediction": predictions[3],
    }


# shot 0 -> train, shot 1 -> test (odd), shot 8 -> validation (even, % 10 == 8)
GOOGLE_SHOTS = [
    _shot(0, b"\x00" * 25, (False, False, False, False), False),
    _shot(1, b"\x81" + b"\x00" * 23 + b"\x03", (True, False, True, False), True),
    _shot(8, b"\x10" + b"\x00" * 24, (False, True, False, True), False),
]

QASMBENCH_CIRCUIT = [{
    "source_record_id": "circ-src-1", "circuit_id": "qec_sm_n5:source",
    "benchmark_name": "qec_sm_n5", "variant": "source",
    "register_declarations": json.dumps({
        "qregs": [{"name": "q", "size": 3}, {"name": "a", "size": 2}],
        "cregs": [{"name": "c", "size": 3}, {"name": "syn", "size": 2}],
    }),
    "qubit_count": 5, "measurement_count": 5, "two_qubit_gate_count": 4,
}]
QASMBENCH_CHECKS = [
    {"source_record_id": "chk-0", "circuit_id": "qec_sm_n5:source", "check_id": "a[0]#0",
     "ancilla_qubit": "a[0]", "data_qubits": ["q[0]", "q[1]"], "syndrome_bit": "syn[0]"},
    {"source_record_id": "chk-1", "circuit_id": "qec_sm_n5:source", "check_id": "a[1]#0",
     "ancilla_qubit": "a[1]", "data_qubits": ["q[1]", "q[2]"], "syndrome_bit": "syn[1]"},
]
QASMBENCH_CORRECTIONS = [
    {"source_record_id": f"cor-{value}", "circuit_id": "qec_sm_n5:source", "condition_register": "syn",
     "condition_value": value, "gate": "x", "target_qubit": target}
    for value, target in ((1, "q[0]"), (2, "q[2]"), (3, "q[1]"))
]


def _fake_silver(google_shots=GOOGLE_SHOTS):
    def read(_settings, object_name: str) -> tuple[pa.Table, str]:
        if "qec_syndromes" in object_name:
            table = pa.Table.from_pylist(SYNDROME_ROWS, schema=SYNDROMES_SILVER_SCHEMA)
        elif "stabilizer_check" in object_name:
            table = pa.Table.from_pylist(QASMBENCH_CHECKS)
        elif "conditional_correction" in object_name:
            table = pa.Table.from_pylist(QASMBENCH_CORRECTIONS)
        elif "qasmbench" in object_name:
            table = pa.Table.from_pylist(QASMBENCH_CIRCUIT)
        elif "experiment" in object_name:
            table = pa.Table.from_pylist(GOOGLE_EXPERIMENT)
        else:
            table = pa.Table.from_pylist(google_shots)
        return table, f"fake-hash-{object_name}"
    return read


# ---------------------------------------------------------------------------
# Fixtures: throwaway schema, Gold loaded through load_postgres.run()
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_schema():
    settings = Settings.from_environment()
    schema_name = f"test_gold_{uuid.uuid4().hex[:8]}"
    try:
        conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    conn.execute(f"CREATE SCHEMA {schema_name}")
    yield settings, schema_name, conn
    conn.execute(f"DROP SCHEMA {schema_name} CASCADE")
    conn.close()


@pytest.fixture
def ml_env(pg_schema, monkeypatch, tmp_path):
    """Point both stages at the throwaway schema and tmp results; capture lake writes."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "GOLD_SCHEMA", schema_name)
    monkeypatch.setattr(lp.connections, "postgres_connection", lambda _s: psycopg.connect(settings.postgres_dsn))

    results_root = tmp_path / "results" / "part1"
    (results_root / "analysis").mkdir(parents=True)
    for sql_file in ANALYSIS_SQL_DIR.glob("query*.sql"):
        shutil.copy(sql_file, results_root / "analysis" / sql_file.name)
    monkeypatch.setattr(lp, "RESULTS_PART1", results_root)
    monkeypatch.setattr(bml, "RESULTS_PART1", results_root)

    written: dict[str, bytes] = {}
    monkeypatch.setattr(bml, "_write_lake_object", lambda _s, name, data: written.__setitem__(name, data))
    monkeypatch.setattr(bml, "_data_release", lambda: "test-release v0")
    return {"conn": conn, "schema": schema_name, "results": results_root, "written": written, "monkeypatch": monkeypatch}


def _load_gold(env, google_shots=GOOGLE_SHOTS) -> None:
    env["monkeypatch"].setattr(lp, "_read_silver_parquet_with_hash", _fake_silver(google_shots))
    lp.run("gold-for-ml-test")


def _written_table(env, path: str) -> pa.Table:
    return pq.read_table(io.BytesIO(env["written"][path]))


# ---------------------------------------------------------------------------
# Full stage against Gold
# ---------------------------------------------------------------------------


def test_run_writes_both_tables_with_the_exact_contract_schema(ml_env) -> None:
    _load_gold(ml_env)
    result = bml.run("ml-test")

    syndrome = _written_table(ml_env, bml.SYNDROME_ML_PATH)
    google = _written_table(ml_env, bml.GOOGLE_ML_PATH)
    assert syndrome.schema.equals(bml.SYNDROME_ML_SCHEMA, check_metadata=False)
    assert google.schema.equals(bml.GOOGLE_ML_SCHEMA, check_metadata=False)
    assert (syndrome.num_rows, google.num_rows) == (4, 3)
    assert result.output_count == 7
    assert sorted(set(syndrome["data_split"].to_pylist())) == ["test", "train", "validation"]
    assert dict(zip(google["shot_index"].to_pylist(), google["data_split"].to_pylist())) == {
        0: "train", 1: "test", 8: "validation",
    }


def test_run_records_hashes_counts_and_release(ml_env) -> None:
    _load_gold(ml_env)
    bml.run("ml-test")

    run_json = json.loads((ml_env["results"] / "run.json").read_text())["build_ml_tables"]
    for path in (bml.SYNDROME_ML_PATH, bml.GOOGLE_ML_PATH):
        assert run_json["output_hashes"][path] == hashlib.sha256(ml_env["written"][path]).hexdigest()
    assert run_json["data_release"] == "test-release v0"
    assert run_json["split_counts"][bml.GOOGLE_ML_PATH] == {"train": 1, "validation": 1, "test": 1}

    row_counts = json.loads((ml_env["results"] / "row_counts.json").read_text())["build_ml_tables"]
    assert row_counts["ml_syndrome_decoder_example"] == {
        "gold_rows": 4, "ml_rows": 4, "unresolved_example_ids": 0, "match": True,
    }
    assert row_counts["ml_google_decoder_example"]["match"] is True


def test_ml_values_are_gold_values_and_packed_bytes_are_reproduced(ml_env) -> None:
    """detector_bits must come back byte-for-byte from Gold (no Bronze/Silver
    re-read), the decoder pivot must restore each shot's own four predictions,
    and quantity must become sample_weight."""
    _load_gold(ml_env)
    bml.run("ml-test")

    google = {r["shot_index"]: r for r in _written_table(ml_env, bml.GOOGLE_ML_PATH).to_pylist()}
    for shot in GOOGLE_SHOTS:
        row = google[shot["shot_index"]]
        assert row["detector_bits"] == shot["detector_bits"]
        assert row["detector_event_count"] == shot["detector_event_count"]
        assert row["actual_observable_flip"] == shot["actual_observable_flip"]
        for column in bml.GOOGLE_META_PREDICTION_COLUMNS:
            assert row[column] == shot[column]

    syndrome = {r["example_id"]: r for r in _written_table(ml_env, bml.SYNDROME_ML_PATH).to_pylist()}
    for silver in SYNDROME_ROWS:
        example_id = lp._syndrome_observation_id(
            silver["experiment_id"], silver["syndrome_bits"], silver["logical_error_label"]
        )
        assert syndrome[example_id]["sample_weight"] == silver["quantity"]
        assert syndrome[example_id]["syndrome_bits"] == silver["syndrome_bits"]


def test_every_example_id_resolves_to_its_gold_record(ml_env) -> None:
    _load_gold(ml_env)
    bml.run("ml-test")
    conn, schema = ml_env["conn"], ml_env["schema"]

    for example_id in _written_table(ml_env, bml.SYNDROME_ML_PATH)["example_id"].to_pylist():
        row = conn.execute(
            f"SELECT source_record_id FROM {schema}.gold_syndrome_observation WHERE observation_id = %s",
            (example_id,),
        ).fetchone()
        assert row is not None

    for example_id in _written_table(ml_env, bml.GOOGLE_ML_PATH)["example_id"].to_pylist():
        row = conn.execute(
            f"SELECT s.source_record_id FROM {schema}.v_ml_google_decoder_example v "
            f"JOIN {schema}.gold_google_shot s USING (experiment_id, shot_index) WHERE v.example_id = %s",
            (example_id,),
        ).fetchone()
        assert row is not None and row[0].startswith("shot-src-")


def test_run_never_reads_bronze_or_silver(ml_env) -> None:
    _load_gold(ml_env)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("build_ml_tables must only read Gold")

    ml_env["monkeypatch"].setattr(lp, "_read_silver_parquet_with_hash", _forbidden)
    ml_env["monkeypatch"].setattr(lp, "_read_silver_parquet", _forbidden)
    ml_env["monkeypatch"].setattr(prepare_data, "_read_bronze_object", _forbidden)
    bml.run("ml-test")
    assert set(ml_env["written"]) == {bml.SYNDROME_ML_PATH, bml.GOOGLE_ML_PATH}


def test_rerun_on_unchanged_gold_writes_byte_identical_tables(ml_env) -> None:
    _load_gold(ml_env)
    bml.run("ml-test-1")
    first = dict(ml_env["written"])
    bml.run("ml-test-2")
    assert ml_env["written"] == first


def test_run_writes_the_three_analysis_results(ml_env) -> None:
    _load_gold(ml_env)
    bml.run("ml-test")

    analysis = ml_env["results"] / "analysis"
    results = sorted(p.name for p in analysis.glob("*_result.csv"))
    assert results == [
        "query1_weighted_syndrome_by_fault_rate_result.csv",
        "query2_google_decoder_error_by_distance_and_location_result.csv",
        "query3_qasmbench_circuit_mapping_result.csv",
    ]
    # query 3: 1 circuit x 2 checks x 3 corrections
    query3 = (analysis / "query3_qasmbench_circuit_mapping_result.csv").read_text().splitlines()
    assert len(query3) == 1 + 6
    # query 2: 1 experiment x 4 decoders
    query2 = (analysis / "query2_google_decoder_error_by_distance_and_location_result.csv").read_text().splitlines()
    assert len(query2) == 1 + 4


def test_contract_failure_writes_nothing(ml_env) -> None:
    """A detector_event_count that disagrees with the bits passes Gold's
    CHECK (it is still <= detector_count) but must be caught by the ML
    contract, before any Parquet or run.json is written."""
    bad_shots = [{**GOOGLE_SHOTS[0], "detector_event_count": 5}, *GOOGLE_SHOTS[1:]]
    _load_gold(ml_env, google_shots=bad_shots)

    with pytest.raises(bml.MLContractError, match="detector_event_count"):
        bml.run("ml-test")
    assert ml_env["written"] == {}
    assert "build_ml_tables" not in json.loads((ml_env["results"] / "run.json").read_text())


# ---------------------------------------------------------------------------
# Contract checks on their own (no database)
# ---------------------------------------------------------------------------


def _syndrome_table(**overrides) -> pa.Table:
    rows = [
        {"example_id": f"id-{split}", "experiment_id": "e", "physical_fault_rate": 0.001,
         "syndrome_bits": PATTERN, "round_count": 4, "check_count": 4,
         "logical_error_label": False, "sample_weight": 10, "data_split": split}
        for split in ("train", "validation", "test")
    ]
    rows[0].update(overrides)
    return pa.Table.from_pylist(rows, schema=bml.SYNDROME_ML_SCHEMA)


def _google_table(**overrides) -> pa.Table:
    rows = []
    for index, split in ((0, "train"), (8, "validation"), (1, "test")):
        rows.append({
            "example_id": f"id-{index}", "experiment_id": "e", "shot_index": index,
            "distance": 3, "rounds": 25, "center_row": 3, "center_col": 5,
            "detector_count": 200, "detector_event_count": 1, "detector_bits": b"\x01" + b"\x00" * 24,
            **{column: False for column in bml.GOOGLE_META_PREDICTION_COLUMNS},
            "actual_observable_flip": False, "data_split": split,
        })
    rows[0].update(overrides)
    return pa.Table.from_pylist(rows, schema=bml.GOOGLE_ML_SCHEMA)


def test_valid_tables_pass_both_contracts() -> None:
    bml.check_syndrome_contract(_syndrome_table())
    bml.check_google_contract(_google_table())


def test_contract_rejects_duplicate_example_ids() -> None:
    with pytest.raises(bml.MLContractError, match="not unique"):
        bml.check_syndrome_contract(_syndrome_table(example_id="id-test"))


def test_contract_rejects_an_empty_split() -> None:
    table = _google_table()
    without_validation = table.filter(pc.not_equal(table["data_split"], "validation"))
    with pytest.raises(bml.MLContractError, match="'validation' is empty"):
        bml.check_google_contract(without_validation)


def test_contract_rejects_a_syndrome_without_16_binary_values() -> None:
    with pytest.raises(bml.MLContractError, match="16 binary values"):
        bml.check_syndrome_contract(_syndrome_table(syndrome_bits=bytes(15)))
    with pytest.raises(bml.MLContractError, match="16 binary values"):
        bml.check_syndrome_contract(_syndrome_table(syndrome_bits=bytes([2] + [0] * 15)))


def test_contract_rejects_wrong_byte_length() -> None:
    with pytest.raises(bml.MLContractError, match="expected ceil"):
        bml.check_google_contract(_google_table(detector_bits=b"\x01" + b"\x00" * 23))


def test_contract_rejects_nonzero_padding_bits() -> None:
    # 10 detectors -> 2 bytes; bits 10..15 of the little-endian second byte are padding
    with pytest.raises(bml.MLContractError, match="padding"):
        bml.check_google_contract(_google_table(
            detector_count=10, detector_bits=b"\x01\x04", detector_event_count=2,
        ))


def test_contract_rejects_event_count_not_matching_set_bits() -> None:
    with pytest.raises(bml.MLContractError, match="number of set bits"):
        bml.check_google_contract(_google_table(detector_event_count=2))


def test_contract_rejects_a_missing_decoder_prediction() -> None:
    with pytest.raises(bml.MLContractError, match="pymatching_prediction has 1 null"):
        bml.check_google_contract(_google_table(pymatching_prediction=None))
