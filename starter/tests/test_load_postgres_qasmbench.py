"""Tests for the Gold load_postgres stage (qasmbench source).

Runs against the real Postgres instance, but entirely inside a throwaway
schema (test_gold_<random>) so these tests never touch the real `gold`
schema's gold_qasmbench_* tables. The schema is created before each test and
dropped afterward via a fixture, regardless of pass/fail. No MinIO/Bronze/
Silver is touched - the Silver tables these tests load from are built
in-memory as small PyArrow tables, not read from the real lake.

Mirrors test_load_postgres.py's structure and coverage for google_qec: same
pg_schema fixture, same test groups (schema creation, foreign keys, value
checks, load-function counts/hashes/tracing, idempotency, all-or-nothing
rollback, Silver/Gold mismatch detection), plus qasmbench-specific scenarios
google_qec has no equivalent of: register decomposition (one row per
qreg/creg) and the four-table TRUNCATE ... CASCADE (one parent, three
dependents).
"""

from __future__ import annotations

import json
import uuid

import psycopg
import pyarrow as pa
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.stages import load_postgres as lp


@pytest.fixture
def pg_schema():
    """Create a uniquely-named throwaway schema, point the connection's
    search_path at it, and drop it afterward - so tests never touch the
    real `gold` schema's gold_qasmbench_* tables."""
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


# qec_sm_n5-shaped Silver rows: the one QASMBench benchmark with an explicit
# parity-check/correction structure (see docs/decision-log.md's qasmbench
# section) - 1 circuit, 4 declared registers, 2 stabilizer checks, 3
# conditional corrections.
ONE_CIRCUIT = [{
    "source_record_id": "circ-src-1", "circuit_id": "qec_sm_n5:source",
    "benchmark_name": "qec_sm_n5", "variant": "source",
    "register_declarations": json.dumps({
        "qregs": [{"name": "q", "size": 3}, {"name": "a", "size": 2}],
        "cregs": [{"name": "c", "size": 3}, {"name": "syn", "size": 2}],
    }),
    "qubit_count": 5, "measurement_count": 5, "two_qubit_gate_count": 4,
}]
TWO_CHECKS = [
    {
        "source_record_id": "circ-src-1::stabilizer::a[0]#0", "circuit_id": "qec_sm_n5:source",
        "check_id": "a[0]#0", "ancilla_qubit": "a[0]", "data_qubits": ["q[0]", "q[1]"], "syndrome_bit": "syn[0]",
    },
    {
        "source_record_id": "circ-src-1::stabilizer::a[1]#0", "circuit_id": "qec_sm_n5:source",
        "check_id": "a[1]#0", "ancilla_qubit": "a[1]", "data_qubits": ["q[1]", "q[2]"], "syndrome_bit": "syn[1]",
    },
]
THREE_CORRECTIONS = [
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


def _fake_read_with_hash(_settings, object_name: str) -> tuple[pa.Table, str]:
    if "stabilizer_check" in object_name:
        table = pa.Table.from_pylist(TWO_CHECKS)
    elif "conditional_correction" in object_name:
        table = pa.Table.from_pylist(THREE_CORRECTIONS)
    else:
        table = pa.Table.from_pylist(ONE_CIRCUIT)  # "circuit.parquet"
    return table, f"fake-hash-{object_name}"


def _fake_read_with_hash_empty(_settings, object_name: str) -> tuple[pa.Table, str]:
    if "stabilizer_check" in object_name:
        table = pa.Table.from_pylist([], schema=pa.Table.from_pylist(TWO_CHECKS).schema)
    elif "conditional_correction" in object_name:
        table = pa.Table.from_pylist([], schema=pa.Table.from_pylist(THREE_CORRECTIONS).schema)
    else:
        table = pa.Table.from_pylist([], schema=pa.Table.from_pylist(ONE_CIRCUIT).schema)
    return table, f"fake-hash-{object_name}"


def _insert_circuit(cur, **overrides) -> None:
    row = {**ONE_CIRCUIT[0], **overrides}
    cur.execute(
        "INSERT INTO gold_qasmbench_circuit "
        "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            row["circuit_id"], row["source_record_id"], row["benchmark_name"], row["variant"],
            row["qubit_count"], row["measurement_count"], row["two_qubit_gate_count"],
        ),
    )


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


def test_schema_creates_all_four_tables(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
            (schema_name,),
        )
        tables = {row[0] for row in cur.fetchall()}
    assert {
        "gold_qasmbench_circuit", "gold_qasmbench_register",
        "gold_qasmbench_stabilizer_check", "gold_qasmbench_conditional_correction",
    } <= tables


# ---------------------------------------------------------------------------
# Foreign keys
# ---------------------------------------------------------------------------


def test_register_foreign_key_rejects_unknown_circuit(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_register (circuit_id, register_kind, register_name, register_size) "
                "VALUES ('does-not-exist', 'qreg', 'q', 3)"
            )
    conn.rollback()


def test_stabilizer_check_foreign_key_rejects_unknown_circuit(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_stabilizer_check "
                "(source_record_id, circuit_id, check_id, ancilla_qubit, data_qubits, syndrome_bit) "
                "VALUES ('x', 'does-not-exist', 'a[0]#0', 'a[0]', %s, 'syn[0]')",
                (["q[0]", "q[1]"],),
            )
    conn.rollback()


def test_conditional_correction_foreign_key_rejects_unknown_circuit(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_conditional_correction "
                "(source_record_id, circuit_id, condition_register, condition_value, gate, target_qubit) "
                "VALUES ('x', 'does-not-exist', 'syn', 1, 'x', 'q[0]')"
            )
    conn.rollback()


# ---------------------------------------------------------------------------
# Value checks and uniqueness rules
# ---------------------------------------------------------------------------


def test_circuit_rejects_nonpositive_qubit_count(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            _insert_circuit(cur, circuit_id="bad:source", source_record_id="x", qubit_count=0)
    conn.rollback()


def test_circuit_rejects_variant_outside_source_or_transpiled(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            _insert_circuit(cur, circuit_id="bad:optimized", source_record_id="x", variant="optimized")
    conn.rollback()


def test_circuit_accepts_both_valid_variants(pg_schema) -> None:
    """'source' and 'transpiled' must both be accepted - the valid side of
    the same constraint as the test above."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        _insert_circuit(cur, circuit_id="qec_sm_n5:source", source_record_id="src-1", variant="source")
        _insert_circuit(cur, circuit_id="qec_sm_n5:transpiled", source_record_id="src-2", variant="transpiled")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 2


def test_circuit_rejects_duplicate_benchmark_and_variant(pg_schema) -> None:
    """Two circuit rows can't share the same (benchmark_name, variant) -
    that combination must be unique, independently of circuit_id."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        _insert_circuit(cur, circuit_id="qec_sm_n5:source", source_record_id="src-1")
        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_circuit(cur, circuit_id="qec_sm_n5:duplicate", source_record_id="src-2")
    conn.rollback()


def test_circuit_rejects_duplicate_source_record_id(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        _insert_circuit(cur, circuit_id="qec_sm_n5:source", source_record_id="same-src", variant="source")
        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_circuit(cur, circuit_id="qec_sm_n5:transpiled", source_record_id="same-src", variant="transpiled")
    conn.rollback()


def test_register_rejects_nonpositive_size(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        _insert_circuit(cur, circuit_id="qec_sm_n5:source", source_record_id="x")
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_register (circuit_id, register_kind, register_name, register_size) "
                "VALUES ('qec_sm_n5:source', 'qreg', 'q', 0)"
            )
    conn.rollback()


def test_register_rejects_kind_outside_qreg_or_creg(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        _insert_circuit(cur, circuit_id="qec_sm_n5:source", source_record_id="x")
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_register (circuit_id, register_kind, register_name, register_size) "
                "VALUES ('qec_sm_n5:source', 'qubit', 'q', 3)"
            )
    conn.rollback()


def test_stabilizer_check_rejects_empty_data_qubits(pg_schema) -> None:
    """The cardinality(data_qubits) >= 1 check must reject an empty array -
    the array_length(data_qubits, 1) >= 1 form this replaced silently
    accepted an empty array, since array_length of an empty array is NULL
    and a NULL CHECK expression passes."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        _insert_circuit(cur, circuit_id="qec_sm_n5:source", source_record_id="x")
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_stabilizer_check "
                "(source_record_id, circuit_id, check_id, ancilla_qubit, data_qubits, syndrome_bit) "
                "VALUES ('x', 'qec_sm_n5:source', 'a[0]#0', 'a[0]', %s, 'syn[0]')",
                ([],),
            )
    conn.rollback()


def test_stabilizer_check_rejects_duplicate_check_id_within_circuit(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        _insert_circuit(cur, circuit_id="qec_sm_n5:source", source_record_id="x")
        cur.execute(
            "INSERT INTO gold_qasmbench_stabilizer_check "
            "(source_record_id, circuit_id, check_id, ancilla_qubit, data_qubits, syndrome_bit) "
            "VALUES ('chk-1', 'qec_sm_n5:source', 'a[0]#0', 'a[0]', %s, 'syn[0]')",
            (["q[0]", "q[1]"],),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_stabilizer_check "
                "(source_record_id, circuit_id, check_id, ancilla_qubit, data_qubits, syndrome_bit) "
                "VALUES ('chk-2', 'qec_sm_n5:source', 'a[0]#0', 'a[0]', %s, 'syn[0]')",
                (["q[0]", "q[1]"],),
            )
    conn.rollback()


def test_conditional_correction_rejects_negative_condition_value(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        _insert_circuit(cur, circuit_id="qec_sm_n5:source", source_record_id="x")
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_conditional_correction "
                "(source_record_id, circuit_id, condition_register, condition_value, gate, target_qubit) "
                "VALUES ('x', 'qec_sm_n5:source', 'syn', -1, 'x', 'q[0]')"
            )
    conn.rollback()


# ---------------------------------------------------------------------------
# load_qasmbench_gold: source_record_id tracing, register decomposition,
# counts, hashes
# ---------------------------------------------------------------------------


def test_load_qasmbench_gold_inserts_expected_row_counts(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)

    circuit_count, register_count, check_count, correction_count, circuit_hash, stabilizer_hash, correction_hash = (
        lp.load_qasmbench_gold(conn, settings)
    )
    conn.commit()

    assert (circuit_count, register_count, check_count, correction_count) == (1, 4, 2, 3)
    assert circuit_hash.startswith("fake-hash-")
    assert stabilizer_hash.startswith("fake-hash-")
    assert correction_hash.startswith("fake-hash-")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_register;")
        assert cur.fetchone()[0] == 4
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_stabilizer_check;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_conditional_correction;")
        assert cur.fetchone()[0] == 3


def test_load_qasmbench_gold_preserves_source_record_id_for_tracing(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT source_record_id FROM gold_qasmbench_circuit WHERE circuit_id = 'qec_sm_n5:source';")
        assert cur.fetchone()[0] == "circ-src-1"

        cur.execute("SELECT source_record_id FROM gold_qasmbench_stabilizer_check ORDER BY check_id;")
        assert {row[0] for row in cur.fetchall()} == {
            "circ-src-1::stabilizer::a[0]#0", "circ-src-1::stabilizer::a[1]#0",
        }

        cur.execute("SELECT source_record_id FROM gold_qasmbench_conditional_correction;")
        assert {row[0] for row in cur.fetchall()} == {
            "circ-src-1::correction::0", "circ-src-1::correction::1", "circ-src-1::correction::2",
        }


def test_load_qasmbench_gold_decomposes_registers_correctly(pg_schema, monkeypatch) -> None:
    """register_declarations's JSON text must decompose into one row per
    declared register, with kind/name/size intact."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT register_kind, register_name, register_size FROM gold_qasmbench_register "
            "WHERE circuit_id = 'qec_sm_n5:source' ORDER BY register_kind, register_name;"
        )
        assert sorted(cur.fetchall()) == sorted([
            ("creg", "c", 3), ("creg", "syn", 2), ("qreg", "a", 2), ("qreg", "q", 3),
        ])


def test_load_qasmbench_gold_is_repeatable_without_duplicating(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)

    lp.load_qasmbench_gold(conn, settings)
    conn.commit()
    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_register;")
        assert cur.fetchone()[0] == 4
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_stabilizer_check;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_conditional_correction;")
        assert cur.fetchone()[0] == 3


def test_truncate_cascade_clears_all_dependent_tables(pg_schema, monkeypatch) -> None:
    """TRUNCATE ... CASCADE on gold_qasmbench_circuit must clear all three
    dependent tables too, even when the next load has no circuits at all -
    proving the cascade isn't silently relying on new data overwriting old."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)

    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash_empty)
    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        for table in (
            "gold_qasmbench_circuit", "gold_qasmbench_register",
            "gold_qasmbench_stabilizer_check", "gold_qasmbench_conditional_correction",
        ):
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            assert cur.fetchone()[0] == 0, f"{table} was not cleared by the cascade"


# ---------------------------------------------------------------------------
# All-or-nothing: a failed load must leave the previous complete data intact
# ---------------------------------------------------------------------------


def test_failed_load_keeps_previous_data_intact(pg_schema, monkeypatch) -> None:
    """Load once successfully, then simulate a failure partway through a
    second load (bad Silver data - a negative condition_value, rejected by
    its CHECK after the circuit and checks were already re-inserted). The
    previous complete data must remain - nothing should be half-overwritten."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)

    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    bad_corrections = [{**THREE_CORRECTIONS[0], "condition_value": -1}]

    def _fake_read_bad(_settings, object_name: str) -> tuple[pa.Table, str]:
        if "stabilizer_check" in object_name:
            table = pa.Table.from_pylist(TWO_CHECKS)
        elif "conditional_correction" in object_name:
            table = pa.Table.from_pylist(bad_corrections)
        else:
            table = pa.Table.from_pylist(ONE_CIRCUIT)
        return table, f"fake-hash-{object_name}"

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_bad)

    with pytest.raises(psycopg.errors.CheckViolation):
        lp.load_qasmbench_gold(conn, settings)
    conn.rollback()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_stabilizer_check;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT data_qubits FROM gold_qasmbench_stabilizer_check WHERE check_id = 'a[0]#0';")
        assert cur.fetchone()[0] == ["q[0]", "q[1]"]  # the original value, not overwritten


# ---------------------------------------------------------------------------
# _load_qasmbench: Silver-to-Gold mismatch detection
# ---------------------------------------------------------------------------


def test_load_qasmbench_detects_row_count_mismatch_and_raises(pg_schema, monkeypatch) -> None:
    """If load_qasmbench_gold's returned counts ever disagreed with what's
    actually in the database, _load_qasmbench must raise rather than
    silently trust the in-memory counts - mirrors
    test_run_detects_row_count_mismatch_and_rolls_back for google_qec."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    real_load_qasmbench_gold = lp.load_qasmbench_gold

    def _lying_load_qasmbench_gold(conn, settings):
        real = real_load_qasmbench_gold(conn, settings)
        return (real[0], real[1], real[2] + 1, real[3], real[4], real[5], real[6])  # lie: claim one extra check

    monkeypatch.setattr(lp, "load_qasmbench_gold", _lying_load_qasmbench_gold)

    with pytest.raises(RuntimeError, match="do not match Silver"):
        lp._load_qasmbench(conn, settings)
    conn.rollback()


# ---------------------------------------------------------------------------
# Shared QEC vocabulary: code_family / distance labels
# ---------------------------------------------------------------------------


def test_load_labels_qec_sm_n5_as_distance_three_repetition_code(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_with_hash)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT code_family, distance FROM gold_qasmbench_circuit WHERE circuit_id = 'qec_sm_n5:source';")
        assert cur.fetchone() == ("repetition_code", 3)


def test_load_leaves_circuits_without_explicit_code_unlabelled(pg_schema, monkeypatch) -> None:
    """A benchmark with no explicit parity-check structure (e.g.
    error_correctiond3_n5) must get NULL, not a guessed label."""
    settings, schema_name, conn = pg_schema
    unlabelled_circuit = [{
        **ONE_CIRCUIT[0], "source_record_id": "circ-src-2", "circuit_id": "qec_en_n5:source",
        "benchmark_name": "qec_en_n5",
    }]

    def _fake_read(_settings, object_name: str) -> tuple[pa.Table, str]:
        if "stabilizer_check" in object_name:
            table = pa.Table.from_pylist([], schema=pa.Table.from_pylist(TWO_CHECKS).schema)
        elif "conditional_correction" in object_name:
            table = pa.Table.from_pylist([], schema=pa.Table.from_pylist(THREE_CORRECTIONS).schema)
        else:
            table = pa.Table.from_pylist(unlabelled_circuit)
        return table, f"fake-hash-{object_name}"

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT code_family, distance FROM gold_qasmbench_circuit;")
        assert cur.fetchone() == (None, None)


def test_load_rejects_repetition_code_label_that_disagrees_with_silver(pg_schema, monkeypatch) -> None:
    """qec_sm_n5 is labelled distance 3 (2 checks over 3 data qubits); if
    Silver ever had only one check for it, the load must raise instead of
    storing a label the parsed circuit does not support."""
    settings, schema_name, conn = pg_schema

    def _fake_read_one_check(_settings, object_name: str) -> tuple[pa.Table, str]:
        if "stabilizer_check" in object_name:
            return pa.Table.from_pylist(TWO_CHECKS[:1]), f"fake-hash-{object_name}"
        return _fake_read_with_hash(_settings, object_name)

    monkeypatch.setattr(lp, "_read_silver_parquet_with_hash", _fake_read_one_check)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
    with pytest.raises(ValueError, match="repetition_code distance 3"):
        lp.load_qasmbench_gold(conn, settings)
    conn.rollback()


def test_circuit_rejects_code_family_without_distance(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_circuit "
                "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, "
                " two_qubit_gate_count, code_family, distance) "
                "VALUES ('x:source', 'x', 'x', 'source', 5, 5, 4, 'repetition_code', NULL)"
            )
