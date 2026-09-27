"""Tests for the qec_syndromes Gold-loading logic in load_postgres.py.

These tests call create_syndrome_tables()/load_syndrome_tables() directly --
never run(), which is hardcoded to the real `gold` schema everyone else's
tables also live in. Instead, each test gets its own uniquely-named,
disposable schema, which is dropped afterwards whether the test passes or
fails. This guarantees these tests never read, write, or drop anything
belonging to another source's Gold tables.
"""

from __future__ import annotations

import uuid

import pyarrow as pa
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection
from quantum_lake_student.stages import load_postgres as lp
from quantum_lake_student.stages.prepare_data import SYNDROMES_SILVER_SCHEMA

ROW_A = {
    "source_record_id": "rec-a",
    "experiment_id": "exp-1",
    "physical_fault_rate": 0.0001,
    "syndrome_bits": bytes([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "round_count": 4,
    "check_count": 4,
    "logical_error_label": False,
    "quantity": 100,
}
ROW_B_SAME_PATTERN_OTHER_LABEL = {
    **ROW_A,
    "source_record_id": "rec-b",
    "logical_error_label": True,
    "quantity": 30,
}


def _table(*rows: dict) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=SYNDROMES_SILVER_SCHEMA)


@pytest.fixture
def cursor():
    """A cursor pointed at a brand-new schema, dropped after the test."""
    settings = Settings.from_environment()
    schema_name = f"test_gold_syndromes_{uuid.uuid4().hex[:8]}"
    try:
        connection = postgres_connection(settings)
    except Exception:
        pytest.skip("Postgres not reachable")

    try:
        with connection.cursor() as setup_cursor:
            setup_cursor.execute(f"CREATE SCHEMA {schema_name}")
            setup_cursor.execute(f"SET search_path TO {schema_name}")
        connection.commit()

        db_cursor = connection.cursor()
        db_cursor.execute(f"SET search_path TO {schema_name}")
        yield db_cursor
        connection.commit()
    finally:
        with connection.cursor() as teardown_cursor:
            teardown_cursor.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        connection.commit()
        connection.close()


def _count(cursor, table_name: str) -> int:
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]


def test_load_writes_expected_rows_and_keeps_both_labels(cursor):
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A, ROW_B_SAME_PATTERN_OTHER_LABEL))

    assert _count(cursor, "syndrome_experiment") == 1
    assert _count(cursor, "syndrome_pattern") == 1
    assert _count(cursor, "syndrome_observation") == 2

    cursor.execute("SELECT logical_error_label, quantity FROM syndrome_observation ORDER BY quantity")
    rows = cursor.fetchall()
    assert rows == [(True, 30), (False, 100)]


def test_pattern_triggered_check_count_is_computed_correctly(cursor):
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A))

    cursor.execute("SELECT triggered_check_count FROM syndrome_pattern")
    assert cursor.fetchone()[0] == 1  # ROW_A has exactly one set bit


def test_rerun_does_not_duplicate_rows(cursor):
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A, ROW_B_SAME_PATTERN_OTHER_LABEL))
    lp.load_syndrome_tables(cursor, _table(ROW_A, ROW_B_SAME_PATTERN_OTHER_LABEL))

    assert _count(cursor, "syndrome_experiment") == 1
    assert _count(cursor, "syndrome_pattern") == 1
    assert _count(cursor, "syndrome_observation") == 2


def test_rerun_refreshes_quantity_on_the_same_observation(cursor):
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A))

    updated_row = {**ROW_A, "quantity": 999}
    lp.load_syndrome_tables(cursor, _table(updated_row))

    assert _count(cursor, "syndrome_observation") == 1
    cursor.execute("SELECT quantity FROM syndrome_observation")
    assert cursor.fetchone()[0] == 999


def test_observation_rejects_a_pattern_that_was_never_inserted(cursor):
    lp.create_syndrome_tables(cursor)
    cursor.execute(
        """
        INSERT INTO syndrome_experiment (experiment_id, physical_fault_rate, round_count, check_count)
        VALUES ('exp-1', 0.0001, 4, 4)
        """
    )
    with pytest.raises(Exception, match="foreign key|violates"):
        cursor.execute(
            """
            INSERT INTO syndrome_observation
                (observation_id, experiment_id, syndrome_bits, logical_error_label, quantity, source_record_id)
            VALUES ('obs-1', 'exp-1', %s, false, 1, 'rec-x')
            """,
            (bytes(16),),
        )
    cursor.connection.rollback()
