"""Tests for the Gold load_postgres stage (qasmbench source).

Runs against the real Postgres instance, but entirely inside a throwaway
schema (test_gold_<random>) so these tests never touch the real gold_*
tables. The schema is created before each test and dropped afterward via a
fixture, regardless of pass/fail. No MinIO/Bronze/Silver is touched - the
Silver tables these tests load from are built in-memory as small PyArrow
tables, not read from the real lake.

Mirrors test_load_postgres_google.py's structure and coverage exactly (same
pg_schema fixture, same test groups - schema creation, foreign keys, value
checks, load-function counts/tracing, idempotency, all-or-nothing rollback,
full run_qasmbench_gold()), plus qasmbench-specific scenarios the Google
suite has no equivalent of: register decomposition (one row per qreg/creg)
and the four-table TRUNCATE ... CASCADE (one parent, three dependents).
"""

from __future__ import annotations

import json
import uuid

import psycopg
import pyarrow as pa
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.stages import load_postgres as lp


def _postgres_available(settings: Settings) -> bool:
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


needs_postgres = pytest.mark.skipif(
    not _postgres_available(Settings.from_environment()),
    reason="PostgreSQL is not reachable; run `make up` first",
)


@pytest.fixture
def pg_schema():
    """Create a uniquely-named throwaway schema, point the connection's
    search_path at it, and drop it afterward - so tests never touch the
    real 'public' gold_qasmbench_* tables."""
    settings = Settings.from_environment()
    schema_name = f"test_gold_{uuid.uuid4().hex[:8]}"

    conn = psycopg.connect(settings.postgres_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema_name};")
        cur.execute(f"SET search_path TO {schema_name};")
    conn.autocommit = False  # tests need real transactions to test rollback

    yield settings, schema_name, conn

    conn.rollback()  # clean up any uncommitted state before dropping
    conn.autocommit = True  # DROP SCHEMA needs autocommit again
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


def _fake_read(_settings, object_name: str) -> pa.Table:
    if "stabilizer_check" in object_name:
        return pa.Table.from_pylist(TWO_CHECKS)
    if "conditional_correction" in object_name:
        return pa.Table.from_pylist(THREE_CORRECTIONS)
    return pa.Table.from_pylist(ONE_CIRCUIT)  # "circuit.parquet"


def _fake_read_empty(_settings, object_name: str) -> pa.Table:
    if "stabilizer_check" in object_name:
        return pa.Table.from_pylist([], schema=pa.Table.from_pylist(TWO_CHECKS).schema)
    if "conditional_correction" in object_name:
        return pa.Table.from_pylist([], schema=pa.Table.from_pylist(THREE_CORRECTIONS).schema)
    return pa.Table.from_pylist([], schema=pa.Table.from_pylist(ONE_CIRCUIT).schema)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


@needs_postgres
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


@needs_postgres
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


@needs_postgres
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


@needs_postgres
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


@needs_postgres
def test_circuit_rejects_nonpositive_qubit_count(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_circuit "
                "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
                "VALUES ('bad:source', 'x', 'bad', 'source', 0, 5, 4)"
            )
    conn.rollback()


@needs_postgres
def test_circuit_rejects_variant_outside_source_or_transpiled(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_circuit "
                "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
                "VALUES ('bad:optimized', 'x', 'bad', 'optimized', 5, 5, 4)"
            )
    conn.rollback()


@needs_postgres
def test_circuit_accepts_both_valid_variants(pg_schema) -> None:
    """'source' and 'transpiled' must both be accepted - the valid side of
    the same constraint as the test above."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:source', 'src-1', 'qec_sm_n5', 'source', 5, 5, 4)"
        )
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:transpiled', 'src-2', 'qec_sm_n5', 'transpiled', 5, 5, 4)"
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 2


@needs_postgres
def test_circuit_rejects_duplicate_benchmark_and_variant(pg_schema) -> None:
    """Two circuit rows can't share the same (benchmark_name, variant) -
    that combination must be unique, independently of circuit_id."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:source', 'src-1', 'qec_sm_n5', 'source', 5, 5, 4)"
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_circuit "
                "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
                "VALUES ('qec_sm_n5:duplicate', 'src-2', 'qec_sm_n5', 'source', 5, 5, 4)"
            )
    conn.rollback()


@needs_postgres
def test_circuit_rejects_duplicate_source_record_id(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:source', 'same-src', 'qec_sm_n5', 'source', 5, 5, 4)"
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_circuit "
                "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
                "VALUES ('qec_sm_n5:transpiled', 'same-src', 'qec_sm_n5', 'transpiled', 5, 5, 4)"
            )
    conn.rollback()


@needs_postgres
def test_register_rejects_nonpositive_size(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:source', 'x', 'qec_sm_n5', 'source', 5, 5, 4)"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_register (circuit_id, register_kind, register_name, register_size) "
                "VALUES ('qec_sm_n5:source', 'qreg', 'q', 0)"
            )
    conn.rollback()


@needs_postgres
def test_register_rejects_kind_outside_qreg_or_creg(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:source', 'x', 'qec_sm_n5', 'source', 5, 5, 4)"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_register (circuit_id, register_kind, register_name, register_size) "
                "VALUES ('qec_sm_n5:source', 'qubit', 'q', 3)"
            )
    conn.rollback()


@needs_postgres
def test_stabilizer_check_rejects_empty_data_qubits(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:source', 'x', 'qec_sm_n5', 'source', 5, 5, 4)"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_stabilizer_check "
                "(source_record_id, circuit_id, check_id, ancilla_qubit, data_qubits, syndrome_bit) "
                "VALUES ('x', 'qec_sm_n5:source', 'a[0]#0', 'a[0]', %s, 'syn[0]')",
                ([],),
            )
    conn.rollback()


@needs_postgres
def test_stabilizer_check_rejects_duplicate_check_id_within_circuit(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:source', 'x', 'qec_sm_n5', 'source', 5, 5, 4)"
        )
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


@needs_postgres
def test_conditional_correction_rejects_negative_condition_value(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)
        cur.execute(
            "INSERT INTO gold_qasmbench_circuit "
            "(circuit_id, source_record_id, benchmark_name, variant, qubit_count, measurement_count, two_qubit_gate_count) "
            "VALUES ('qec_sm_n5:source', 'x', 'qec_sm_n5', 'source', 5, 5, 4)"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO gold_qasmbench_conditional_correction "
                "(source_record_id, circuit_id, condition_register, condition_value, gate, target_qubit) "
                "VALUES ('x', 'qec_sm_n5:source', 'syn', -1, 'x', 'q[0]')"
            )
    conn.rollback()


# ---------------------------------------------------------------------------
# load_qasmbench_gold: source_record_id tracing, register decomposition, counts
# ---------------------------------------------------------------------------


@needs_postgres
def test_load_qasmbench_gold_inserts_expected_row_counts(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)

    circuit_count, register_count, check_count, correction_count = lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    assert (circuit_count, register_count, check_count, correction_count) == (1, 4, 2, 3)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_register;")
        assert cur.fetchone()[0] == 4
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_stabilizer_check;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_conditional_correction;")
        assert cur.fetchone()[0] == 3


@needs_postgres
def test_load_qasmbench_gold_preserves_source_record_id_for_tracing(pg_schema, monkeypatch) -> None:
    """Every Gold circuit/check/correction row must carry the same
    source_record_id that was on its Silver row, so a Gold row can be traced
    back."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

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


@needs_postgres
def test_load_qasmbench_gold_decomposes_registers_correctly(pg_schema, monkeypatch) -> None:
    """register_declarations's JSON text must decompose into one row per
    declared register, with kind/name/size intact - the concrete check
    behind the register-decomposition decision documented in
    docs/decision-log.md."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

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


@needs_postgres
def test_load_qasmbench_gold_is_repeatable_without_duplicating(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

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


@needs_postgres
def test_truncate_cascade_clears_all_dependent_tables(pg_schema, monkeypatch) -> None:
    """TRUNCATE ... CASCADE on gold_qasmbench_circuit must clear all three
    dependent tables too, even when the next load has no circuits at all -
    proving the cascade isn't silently relying on the new data happening to
    overwrite the old."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)

    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read_empty)
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


@needs_postgres
def test_failed_load_keeps_previous_data_intact(pg_schema, monkeypatch) -> None:
    """Load once successfully, then simulate a failure partway through a
    second load (bad Silver data - an empty data_qubits array). The previous
    complete data must remain - nothing should be half-overwritten."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.QASMBENCH_GOLD_SCHEMA_SQL)

    # first, good load
    lp.load_qasmbench_gold(conn, settings)
    conn.commit()

    # second attempt: Silver data now violates a constraint (empty data_qubits) -
    # load_qasmbench_gold should raise, and the caller is expected to roll back
    bad_checks = [{**TWO_CHECKS[0], "data_qubits": []}]

    def _fake_read_bad(_settings, object_name: str) -> pa.Table:
        if "stabilizer_check" in object_name:
            return pa.Table.from_pylist(bad_checks)
        if "conditional_correction" in object_name:
            return pa.Table.from_pylist(THREE_CORRECTIONS)
        return pa.Table.from_pylist(ONE_CIRCUIT)

    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read_bad)

    with pytest.raises(psycopg.errors.CheckViolation):
        lp.load_qasmbench_gold(conn, settings)
    conn.rollback()  # this is what run_qasmbench_gold's `with` block does on failure

    # previous good data must still be there, untouched
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_stabilizer_check;")
        assert cur.fetchone()[0] == 2
        cur.execute(
            "SELECT data_qubits FROM gold_qasmbench_stabilizer_check WHERE check_id = 'a[0]#0';"
        )
        assert cur.fetchone()[0] == ["q[0]", "q[1]"]  # the original value, not overwritten


# ---------------------------------------------------------------------------
# Full run_qasmbench_gold(): schema + load + evidence files, in the isolated
# test schema
# ---------------------------------------------------------------------------


@needs_postgres
def test_run_qasmbench_gold_writes_gold_and_run_evidence(pg_schema, monkeypatch, tmp_path) -> None:
    """Runs the full run_qasmbench_gold() against the throwaway schema.
    run_qasmbench_gold() calls connections.postgres_connection() internally,
    so we redirect it to a freshly-opened connection whose search_path is
    already the test schema."""
    settings, schema_name, _fixture_conn = pg_schema

    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    def _test_postgres_connection(_settings):
        c = psycopg.connect(settings.postgres_dsn)
        with c.cursor() as cur:
            cur.execute(f"SET search_path TO {schema_name};")
        return c

    monkeypatch.setattr(lp.connections, "postgres_connection", _test_postgres_connection)
    results_root = tmp_path / "results" / "part1"
    monkeypatch.setattr(lp, "RESULTS_PART1", results_root)

    result = lp.run_qasmbench_gold(settings, "gold-test-run")

    assert (result.stage, result.output_count) == ("load_postgres:qasmbench", 10)  # 1+4+2+3
    assert (results_root / "run.json").exists()
    assert (results_root / "row_counts.json").exists()

    with _fixture_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_register;")
        assert cur.fetchone()[0] == 4
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_stabilizer_check;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_conditional_correction;")
        assert cur.fetchone()[0] == 3
