"""Tests for the Gold load_postgres stage (google_qec source).

Runs against the real Postgres instance, but entirely inside a throwaway
schema (test_gold_<random>) so these tests never touch the real gold_*
tables. The schema is created before each test and dropped afterward via a
fixture, regardless of pass/fail. No MinIO/Bronze/Silver is touched - the
Silver tables these tests load from are built in-memory as small PyArrow
tables, not read from the real lake.

Covers: table creation, source_record_id tracing and its UNIQUE constraint,
the decoder/decoder_prediction tables, the composite foreign key that ties
each shot's copied measurement_count/detector_count back to its experiment,
exact per-experiment byte-length checks, detector_event_count <= detector_count,
the basis-inclusive processor-location uniqueness rule, row-count
reconciliation, and the all-or-nothing rollback guarantee.

run() loads google_qec, qec_syndromes, and qasmbench together in one
transaction, so the run() tests also feed it small in-memory syndromes and
qasmbench tables and point its GOLD_SCHEMA at the throwaway test schema.
"""

from __future__ import annotations

import json
import uuid

import psycopg
import pyarrow as pa
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.stages import load_postgres as lp
from quantum_lake_student.stages.prepare_data import SYNDROMES_SILVER_SCHEMA


@pytest.fixture
def pg_schema():
    """Create a uniquely-named throwaway schema, point the connection's
    search_path at it, and drop it afterward - so tests never touch the
    real 'public' gold_google_* tables."""
    settings = Settings.from_environment()
    schema_name = f"test_gold_{uuid.uuid4().hex[:8]}"

    conn = psycopg.connect(settings.postgres_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema_name};")
        cur.execute(f"SET search_path TO {schema_name};")
    conn.autocommit = False

    yield settings, schema_name, conn

    conn.rollback()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema_name} CASCADE;")
    conn.close()


ONE_EXPERIMENT = [{
    "source_record_id": "exp-src-1", "experiment_id": "exp-1", "basis": "X",
    "distance": 3, "rounds": 25, "shots": 2, "center_row": 3, "center_col": 5,
    "measurement_count": 10, "detector_count": 200,
}]
TWO_SHOTS = [
    {
        "source_record_id": "shot-src-0", "experiment_id": "exp-1", "shot_index": 0,
        "measurement_bits": b"\x00" * 2, "sweep_bits": b"\x00\x00", "detector_bits": b"\x00" * 25,
        "detector_event_count": 0, "actual_observable_flip": False,
        "belief_matching_prediction": False, "correlated_matching_prediction": False,
        "pymatching_prediction": False, "tensor_network_contraction_prediction": False,
    },
    {
        "source_record_id": "shot-src-1", "experiment_id": "exp-1", "shot_index": 1,
        "measurement_bits": b"\xff\x01", "sweep_bits": b"\x01\x00", "detector_bits": b"\x01" + b"\x00" * 24,
        "detector_event_count": 1, "actual_observable_flip": True,
        "belief_matching_prediction": True, "correlated_matching_prediction": False,
        "pymatching_prediction": True, "tensor_network_contraction_prediction": False,
    },
]


def _fake_read(_settings, object_name: str) -> pa.Table:
    return (
        pa.Table.from_pylist(ONE_EXPERIMENT)
        if "experiment" in object_name
        else pa.Table.from_pylist(TWO_SHOTS)
    )


def _fake_read_with_hash(_settings, object_name: str) -> tuple[pa.Table, str]:
    table = _fake_read(_settings, object_name)
    return table, f"fake-hash-{object_name}"


# Minimal qec_syndromes Silver rows, needed because run() loads both sources
SYNDROME_ROWS = [
    {
        "source_record_id": "syn-src-0", "experiment_id": "syn-exp-1", "physical_fault_rate": 0.0001,
        "syndrome_bits": bytes([0] * 6 + [1] + [0] * 9), "round_count": 4, "check_count": 4,
        "logical_error_label": False, "quantity": 100, "distance": 3,
    },
    {
        "source_record_id": "syn-src-1", "experiment_id": "syn-exp-1", "physical_fault_rate": 0.0001,
        "syndrome_bits": bytes([0] * 6 + [1] + [0] * 9), "round_count": 4, "check_count": 4,
        "logical_error_label": True, "quantity": 30, "distance": 3,
    },
]


# Minimal qasmbench Silver rows (qec_sm_n5-shaped), needed because run()
# loads all three sources together
ONE_QASMBENCH_CIRCUIT = [{
    "source_record_id": "circ-src-1", "circuit_id": "qec_sm_n5:source",
    "benchmark_name": "qec_sm_n5", "variant": "source",
    "register_declarations": json.dumps({
        "qregs": [{"name": "q", "size": 3}, {"name": "a", "size": 2}],
        "cregs": [{"name": "c", "size": 3}, {"name": "syn", "size": 2}],
    }),
    "qubit_count": 5, "measurement_count": 5, "two_qubit_gate_count": 4,
}]
TWO_QASMBENCH_CHECKS = [
    {
        "source_record_id": "circ-src-1::stabilizer::a[0]#0", "circuit_id": "qec_sm_n5:source",
        "check_id": "a[0]#0", "ancilla_qubit": "a[0]", "data_qubits": ["q[0]", "q[1]"], "syndrome_bit": "syn[0]",
    },
    {
        "source_record_id": "circ-src-1::stabilizer::a[1]#0", "circuit_id": "qec_sm_n5:source",
        "check_id": "a[1]#0", "ancilla_qubit": "a[1]", "data_qubits": ["q[1]", "q[2]"], "syndrome_bit": "syn[1]",
    },
]
THREE_QASMBENCH_CORRECTIONS = [
    {
        "source_record_id": "circ-src-1::correction::0", "circuit_id": "qec_sm_n5:source",
        "condition_register": "syn", "condition_value": 1, "gate": "x", "target_qubit": "q[0]",
    },
    {
        "source_record_id": "circ-src-1::correction::1", "circuit_id": "qec_sm_n5:source",
        "condition_register": "syn", "condition_value": 2, "gate": "x", "target_qubit": "q[2]",
    },
    {
        "source_record_id": "circ-src-1::correction::2", "circuit_id": "qec_sm_n5:source",
        "condition_register": "syn", "condition_value": 3, "gate": "x", "target_qubit": "q[1]",
    },
]


def _fake_read_all_sources(
    _settings, object_name: str, syndrome_rows=SYNDROME_ROWS, qasmbench_circuits=ONE_QASMBENCH_CIRCUIT
) -> tuple[pa.Table, str]:
    """Fake Silver reader for run(): syndromes for the qec_syndromes path,
    qasmbench for its path, Google otherwise."""
    if "qec_syndromes" in object_name:
        table = pa.Table.from_pylist(syndrome_rows, schema=SYNDROMES_SILVER_SCHEMA)
        return table, f"fake-hash-{object_name}"
    if "qasmbench" in object_name:
        if "stabilizer_check" in object_name:
            table = pa.Table.from_pylist(TWO_QASMBENCH_CHECKS)
        elif "conditional_correction" in object_name:
            table = pa.Table.from_pylist(THREE_QASMBENCH_CORRECTIONS)
        else:
            table = pa.Table.from_pylist(qasmbench_circuits)
        return table, f"fake-hash-{object_name}"
    return _fake_read_with_hash(_settings, object_name)


def _insert_experiment(cur, **overrides) -> None:
    row = {**ONE_EXPERIMENT[0], **overrides}
    cur.execute(
        """
        INSERT INTO gold_google_experiment
            (experiment_id, source_record_id, basis, code_family, distance, rounds, shots, center_row, center_col,
             measurement_count, detector_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            row["experiment_id"], row["source_record_id"], row["basis"], row.get("code_family", "surface_code"),
            row["distance"], row["rounds"],
            row["shots"], row["center_row"], row["center_col"], row["measurement_count"], row["detector_count"],
        ),
    )


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


def test_schema_creates_all_four_tables(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
            (schema_name,),
        )
        tables = {row[0] for row in cur.fetchall()}
    assert {
        "gold_google_experiment", "gold_google_shot",
        "gold_google_decoder", "gold_google_decoder_prediction",
    } <= tables


# ---------------------------------------------------------------------------
# Basic constraints
# ---------------------------------------------------------------------------


def test_experiment_rejects_nonpositive_shots(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            _insert_experiment(cur, experiment_id="exp-bad", source_record_id="x", shots=0)
    conn.rollback()


def test_experiment_rejects_basis_outside_x_or_z(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            _insert_experiment(cur, experiment_id="exp-bad", source_record_id="x", basis="Y")
    conn.rollback()


def test_experiment_rejects_detector_count_not_matching_distance(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            _insert_experiment(cur, experiment_id="exp-bad", source_record_id="x", distance=3, detector_count=600)
    conn.rollback()


def test_experiment_accepts_both_valid_distance_detector_pairs(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-d3", source_record_id="s1", distance=3, detector_count=200)
        _insert_experiment(
            cur, experiment_id="exp-d5", source_record_id="s2",
            distance=5, detector_count=600, center_row=9, center_col=9,
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 2


# ---------------------------------------------------------------------------
# UNIQUE constraints: source_record_id, and basis-inclusive processor location
# ---------------------------------------------------------------------------


def test_experiment_rejects_duplicate_source_record_id(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-a", source_record_id="dup-id")
        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_experiment(
                cur, experiment_id="exp-b", source_record_id="dup-id",
                center_row=9, center_col=9,
            )
    conn.rollback()


def test_shot_rejects_duplicate_source_record_id(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-1", source_record_id="exp-src-1")
        cur.execute(
            """
            INSERT INTO gold_google_shot
                (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                 measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
            VALUES ('exp-1', 'dup-shot', 0, 10, 200, %s, %s, %s, 0, false)
            """,
            (b"\x00" * 2, b"\x00\x00", b"\x00" * 25),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                     measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
                VALUES ('exp-1', 'dup-shot', 1, 10, 200, %s, %s, %s, 0, false)
                """,
                (b"\x00" * 2, b"\x00\x00", b"\x00" * 25),
            )
    conn.rollback()


def test_experiment_rejects_duplicate_location_with_same_basis(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-first", source_record_id="s1")
        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_experiment(cur, experiment_id="exp-duplicate", source_record_id="s2")
    conn.rollback()


def test_experiment_allows_same_location_with_different_basis(pg_schema) -> None:
    """basis is now part of the uniqueness rule - same distance/rounds/location
    but a different basis must be allowed (they are genuinely different experiments)."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-x", source_record_id="s1", basis="X")
        _insert_experiment(cur, experiment_id="exp-z", source_record_id="s2", basis="Z")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 2


# ---------------------------------------------------------------------------
# The composite foreign key: shot's copied counts must match its experiment
# ---------------------------------------------------------------------------


def test_shot_foreign_key_rejects_unknown_experiment(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                     measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
                VALUES ('does-not-exist', 'x', 0, 10, 200, %s, %s, %s, 0, false)
                """,
                (b"\x00" * 2, b"\x00\x00", b"\x00" * 25),
            )
    conn.rollback()


def test_shot_composite_fk_rejects_mismatched_counts(pg_schema) -> None:
    """A shot copying the wrong measurement_count/detector_count for its
    experiment_id must be rejected - the composite FK is what enforces this
    cross-table consistency rule at the row level."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-1", source_record_id="exp-src-1", measurement_count=10, detector_count=200)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                     measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
                VALUES ('exp-1', 'shot-x', 0, 10, 600, %s, %s, %s, 0, false)
                """,
                (b"\x00" * 2, b"\x00\x00", b"\x00" * 75),  # detector_count=600 doesn't match exp-1's 200
            )
    conn.rollback()


def test_shot_composite_fk_accepts_matching_counts(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-1", source_record_id="exp-src-1", measurement_count=10, detector_count=200)
        cur.execute(
            """
            INSERT INTO gold_google_shot
                (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                 measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
            VALUES ('exp-1', 'shot-x', 0, 10, 200, %s, %s, %s, 0, false)
            """,
            (b"\x00" * 2, b"\x00\x00", b"\x00" * 25),
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_shot;")
        assert cur.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Exact per-experiment byte-length checks and detector_event_count bound
# ---------------------------------------------------------------------------


def test_shot_rejects_detector_bits_not_matching_own_detector_count(pg_schema) -> None:
    """detector_bits must be exactly ceil(detector_count/8) bytes for THIS
    shot's own detector_count - not just any of the two known valid lengths."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-1", source_record_id="exp-src-1", measurement_count=10, detector_count=200)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                     measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
                VALUES ('exp-1', 'shot-x', 0, 10, 200, %s, %s, %s, 0, false)
                """,
                (b"\x00" * 2, b"\x00\x00", b"\x00" * 75),  # 75 bytes is valid for d5, not this d3-shaped experiment
            )
    conn.rollback()


def test_shot_rejects_measurement_bits_wrong_byte_length(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-1", source_record_id="exp-src-1", measurement_count=10, detector_count=200)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                     measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
                VALUES ('exp-1', 'shot-x', 0, 10, 200, %s, %s, %s, 0, false)
                """,
                (b"\x00" * 5, b"\x00\x00", b"\x00" * 25),  # measurement_count=10 needs ceil(10/8)=2 bytes, not 5
            )
    conn.rollback()


def test_shot_rejects_detector_event_count_exceeding_detector_count(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        _insert_experiment(cur, experiment_id="exp-1", source_record_id="exp-src-1", measurement_count=10, detector_count=200)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                     measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
                VALUES ('exp-1', 'shot-x', 0, 10, 200, %s, %s, %s, 201, false)
                """,
                (b"\x00" * 2, b"\x00\x00", b"\x00" * 25),  # 201 fired detectors, but only 200 exist
            )
    conn.rollback()


# ---------------------------------------------------------------------------
# load_google_gold: source_record_id tracing, decoder tables, counts, hashes
# ---------------------------------------------------------------------------


def test_load_google_gold_inserts_expected_row_counts(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)

    exp_count, shot_count, decoder_prediction_count, experiment_hash, shot_hash = lp.load_google_gold(conn, settings)
    conn.commit()

    assert (exp_count, shot_count, decoder_prediction_count) == (1, 2, 8)  # 2 shots x 4 decoders
    assert experiment_hash.startswith("fake-hash-")
    assert shot_hash.startswith("fake-hash-")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_google_shot;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_google_decoder;")
        assert cur.fetchone()[0] == 4
        cur.execute("SELECT COUNT(*) FROM gold_google_decoder_prediction;")
        assert cur.fetchone()[0] == 8


def test_load_google_gold_preserves_source_record_id_for_tracing(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
    lp.load_google_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT source_record_id FROM gold_google_experiment WHERE experiment_id = 'exp-1';")
        assert cur.fetchone()[0] == "exp-src-1"

        cur.execute("SELECT shot_index, source_record_id FROM gold_google_shot ORDER BY shot_index;")
        assert cur.fetchall() == [(0, "shot-src-0"), (1, "shot-src-1")]


def test_load_google_gold_labels_experiments_with_shared_vocabulary(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
    lp.load_google_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT code_family, distance FROM gold_google_experiment WHERE experiment_id = 'exp-1';")
        assert cur.fetchone() == ("surface_code", 3)


def test_load_google_gold_copies_experiment_counts_onto_each_shot(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
    lp.load_google_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT measurement_count, detector_count FROM gold_google_shot;")
        rows = cur.fetchall()
    assert all(row == (10, 200) for row in rows)  # matches ONE_EXPERIMENT's declared counts


def test_load_google_gold_decoder_prediction_marks_wrong_correctly(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
    lp.load_google_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT was_wrong FROM gold_google_decoder_prediction "
            "WHERE experiment_id = 'exp-1' AND shot_index = 0 AND decoder_name = 'belief_matching';"
        )
        assert cur.fetchone()[0] is False  # shot 0: actual False, predicted False

        cur.execute(
            "SELECT was_wrong FROM gold_google_decoder_prediction "
            "WHERE experiment_id = 'exp-1' AND shot_index = 1 AND decoder_name = 'correlated_matching';"
        )
        assert cur.fetchone()[0] is True  # shot 1: actual True, predicted False


def test_load_google_gold_is_repeatable_without_duplicating(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)

    lp.load_google_gold(conn, settings)
    conn.commit()
    lp.load_google_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_google_shot;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_google_decoder_prediction;")
        assert cur.fetchone()[0] == 8


# ---------------------------------------------------------------------------
# All-or-nothing: a failed load must leave the previous complete data intact
# ---------------------------------------------------------------------------


def test_failed_load_keeps_previous_data_intact(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)

    lp.load_google_gold(conn, settings)
    conn.commit()

    bad_experiment = [{**ONE_EXPERIMENT[0], "basis": "Y"}]

    def _fake_read_bad(_settings, object_name: str) -> tuple[pa.Table, str]:
        table = (
            pa.Table.from_pylist(bad_experiment)
            if "experiment" in object_name
            else pa.Table.from_pylist(TWO_SHOTS)
        )
        return table, f"fake-hash-{object_name}"

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_bad)

    with pytest.raises(psycopg.errors.CheckViolation):
        lp.load_google_gold(conn, settings)
    conn.rollback()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT basis FROM gold_google_experiment WHERE experiment_id = 'exp-1';")
        assert cur.fetchone()[0] == "X"
        cur.execute("SELECT COUNT(*) FROM gold_google_shot;")
        assert cur.fetchone()[0] == 2


# ---------------------------------------------------------------------------
# Full run(): schema + load + row-count reconciliation + evidence files
# ---------------------------------------------------------------------------


def test_run_writes_gold_and_run_evidence(pg_schema, monkeypatch, tmp_path) -> None:
    settings, schema_name, _fixture_conn = pg_schema

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_all_sources)
    monkeypatch.setattr(lp, "GOLD_SCHEMA", schema_name)

    def _test_postgres_connection(_settings):
        c = psycopg.connect(settings.postgres_dsn)
        with c.cursor() as cur:
            cur.execute(f"SET search_path TO {schema_name};")
        return c

    monkeypatch.setattr(lp.connections, "postgres_connection", _test_postgres_connection)
    results_root = tmp_path / "results" / "part1"
    monkeypatch.setattr(lp, "RESULTS_PART1", results_root)

    result = lp.run("gold-test-run")

    # 1 experiment + 2 shots (google_qec) + 2 observations (qec_syndromes)
    # + 1 circuit + 4 registers + 2 checks + 3 corrections (qasmbench)
    assert (result.stage, result.output_count) == ("load_postgres", 5 + 10)
    assert (results_root / "run.json").exists()
    assert (results_root / "row_counts.json").exists()

    run_json = json.loads((results_root / "run.json").read_text())["load_postgres"]["google_qec"]
    assert "input_hashes" in run_json
    assert run_json["input_hashes"]["silver/google_qec/experiment.parquet"].startswith("fake-hash-")

    row_counts = json.loads((results_root / "row_counts.json").read_text())["load_postgres"]["google_qec"]
    assert row_counts["experiment"]["match"] is True
    assert row_counts["shot"]["match"] is True
    assert row_counts["decoder_prediction"]["match"] is True

    with _fixture_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_google_shot;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_google_decoder_prediction;")
        assert cur.fetchone()[0] == 8
        cur.execute("SELECT COUNT(*) FROM gold_syndrome_observation;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_register;")
        assert cur.fetchone()[0] == 4
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_stabilizer_check;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_conditional_correction;")
        assert cur.fetchone()[0] == 3

        # shared-vocabulary view: one row per experiment/circuit, sources stacked, never joined
        cur.execute(
            "SELECT source_name, entity_kind, entity_id, code_family, distance, rounds "
            "FROM v_qec_experiment_catalog ORDER BY source_name;"
        )
        assert cur.fetchall() == [
            ("google_qec", "experiment", "exp-1", "surface_code", 3, 25),
            ("qasmbench", "circuit", "qec_sm_n5:source", "repetition_code", 3, None),
            ("qec_syndromes", "experiment", "syn-exp-1", "surface_code", 3, 4),
        ]

    run_syndromes = json.loads((results_root / "run.json").read_text())["load_postgres"]["qec_syndromes"]
    assert set(run_syndromes["output_row_counts"]) == {
        "gold_syndrome_experiment", "gold_syndrome_pattern", "gold_syndrome_observation",
    }

    # every source writes its own evidence into the shared results files
    syndromes_counts = json.loads((results_root / "row_counts.json").read_text())["load_postgres"]["qec_syndromes"]
    assert syndromes_counts["rows_match"] is True
    assert syndromes_counts["quantity_match"] is True
    syndromes_run = json.loads((results_root / "run.json").read_text())["load_postgres"]["qec_syndromes"]
    assert syndromes_run["input_hash"].startswith("fake-hash-")

    qasmbench_counts = json.loads((results_root / "row_counts.json").read_text())["load_postgres"]["qasmbench"]
    assert qasmbench_counts["circuit"]["match"] is True
    assert qasmbench_counts["register"]["match"] is True
    assert qasmbench_counts["stabilizer_check"]["match"] is True
    assert qasmbench_counts["conditional_correction"]["match"] is True
    qasmbench_run = json.loads((results_root / "run.json").read_text())["load_postgres"]["qasmbench"]
    assert qasmbench_run["input_hashes"]["silver/qasmbench/circuit.parquet"].startswith("fake-hash-")


def test_run_detects_row_count_mismatch_and_rolls_back(pg_schema, monkeypatch, tmp_path) -> None:
    """If load_google_gold's returned counts ever disagreed with what's
    actually in the database, run() must raise and roll back rather than
    silently trust the in-memory counts. Simulated by monkeypatching
    load_google_gold to lie about its shot count.

    Because run()'s schema-creation and the load both happen inside the
    same transaction, a rollback here undoes CREATE TABLE too - so the
    correct check is that gold_google_experiment does not exist at all
    afterward, not that it exists with zero rows.
    """
    settings, schema_name, _fixture_conn = pg_schema

    real_load_google_gold = lp.load_google_gold  # capture the original before patching, to avoid infinite recursion

    def _lying_load_google_gold(conn, settings):
        real = real_load_google_gold(conn, settings)
        return (real[0], real[1] + 1, real[2], real[3], real[4])  # lie: claim one extra shot

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_all_sources)
    monkeypatch.setattr(lp, "GOLD_SCHEMA", schema_name)
    monkeypatch.setattr(lp, "load_google_gold", _lying_load_google_gold)

    def _test_postgres_connection(_settings):
        c = psycopg.connect(settings.postgres_dsn)
        with c.cursor() as cur:
            cur.execute(f"SET search_path TO {schema_name};")
        return c

    monkeypatch.setattr(lp.connections, "postgres_connection", _test_postgres_connection)
    results_root = tmp_path / "results" / "part1"
    monkeypatch.setattr(lp, "RESULTS_PART1", results_root)

    with pytest.raises(RuntimeError, match="do not match Silver"):
        lp.run("gold-test-mismatch")

    with _fixture_conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = 'gold_google_experiment');",
            (schema_name,),
        )
        table_exists = cur.fetchone()[0]
    assert not table_exists, (
        "the failed run's transaction (schema creation + insert) should have "
        "been fully rolled back, leaving no gold_google_experiment table at all"
    )


def test_run_rolls_back_google_when_syndromes_load_fails(pg_schema, monkeypatch, tmp_path) -> None:
    """Both sources load in one transaction: if the syndromes part fails
    (here: a quantity of 0 violates its CHECK), the Google tables loaded
    earlier in the same run must be rolled back too."""
    settings, schema_name, _fixture_conn = pg_schema

    bad_syndrome_rows = [{**SYNDROME_ROWS[0], "quantity": 0}]

    def _fake_read_bad_syndromes(_settings, object_name: str):
        return _fake_read_all_sources(_settings, object_name, syndrome_rows=bad_syndrome_rows)

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_bad_syndromes)
    monkeypatch.setattr(lp, "GOLD_SCHEMA", schema_name)

    def _test_postgres_connection(_settings):
        return psycopg.connect(settings.postgres_dsn)

    monkeypatch.setattr(lp.connections, "postgres_connection", _test_postgres_connection)
    monkeypatch.setattr(lp, "RESULTS_PART1", tmp_path / "results" / "part1")

    with pytest.raises(psycopg.errors.CheckViolation):
        lp.run("gold-test-syndromes-fail")

    with _fixture_conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = 'gold_google_experiment');",
            (schema_name,),
        )
        assert cur.fetchone()[0] is False
    assert not (tmp_path / "results" / "part1" / "run.json").exists()


def test_run_rolls_back_google_and_syndromes_when_qasmbench_load_fails(pg_schema, monkeypatch, tmp_path) -> None:
    """All three sources load in one transaction: if the qasmbench part
    fails (here: a negative condition_value violates its CHECK), the Google
    and syndromes tables loaded earlier in the same run must be rolled back
    too - mirrors test_run_rolls_back_google_when_syndromes_load_fails."""
    settings, schema_name, _fixture_conn = pg_schema

    bad_qasmbench_corrections = [{**THREE_QASMBENCH_CORRECTIONS[0], "condition_value": -1}]

    def _fake_read_bad_qasmbench(_settings, object_name: str):
        if "qasmbench" in object_name and "conditional_correction" in object_name:
            return pa.Table.from_pylist(bad_qasmbench_corrections), f"fake-hash-{object_name}"
        return _fake_read_all_sources(_settings, object_name)

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_bad_qasmbench)
    monkeypatch.setattr(lp, "GOLD_SCHEMA", schema_name)

    def _test_postgres_connection(_settings):
        return psycopg.connect(settings.postgres_dsn)

    monkeypatch.setattr(lp.connections, "postgres_connection", _test_postgres_connection)
    monkeypatch.setattr(lp, "RESULTS_PART1", tmp_path / "results" / "part1")

    with pytest.raises(psycopg.errors.CheckViolation):
        lp.run("gold-test-qasmbench-fail")

    with _fixture_conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = 'gold_google_experiment');",
            (schema_name,),
        )
        assert cur.fetchone()[0] is False
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = 'gold_syndrome_observation');",
            (schema_name,),
        )
        assert cur.fetchone()[0] is False
    assert not (tmp_path / "results" / "part1" / "run.json").exists()
