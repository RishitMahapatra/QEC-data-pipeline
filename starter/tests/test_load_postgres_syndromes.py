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
    "distance": 3,
}
ROW_B_SAME_PATTERN_OTHER_LABEL = {
    **ROW_A,
    "source_record_id": "rec-b",
    "logical_error_label": True,
    "quantity": 30,
}
DISTINCT_PATTERN_BITS = bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


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

    assert _count(cursor, "gold_syndrome_experiment") == 1
    assert _count(cursor, "gold_syndrome_pattern") == 1
    assert _count(cursor, "gold_syndrome_observation") == 2

    cursor.execute("SELECT logical_error_label, quantity FROM gold_syndrome_observation ORDER BY quantity")
    rows = cursor.fetchall()
    assert rows == [(True, 30), (False, 100)]


def test_pattern_triggered_check_count_is_computed_correctly(cursor):
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A))

    cursor.execute("SELECT triggered_check_count FROM gold_syndrome_pattern")
    assert cursor.fetchone()[0] == 1  # ROW_A has exactly one set bit


def test_rerun_does_not_duplicate_rows(cursor):
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A, ROW_B_SAME_PATTERN_OTHER_LABEL))
    lp.load_syndrome_tables(cursor, _table(ROW_A, ROW_B_SAME_PATTERN_OTHER_LABEL))

    assert _count(cursor, "gold_syndrome_experiment") == 1
    assert _count(cursor, "gold_syndrome_pattern") == 1
    assert _count(cursor, "gold_syndrome_observation") == 2


def test_rerun_refreshes_quantity_on_the_same_observation(cursor):
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A))

    updated_row = {**ROW_A, "quantity": 999}
    lp.load_syndrome_tables(cursor, _table(updated_row))

    assert _count(cursor, "gold_syndrome_observation") == 1
    cursor.execute("SELECT quantity FROM gold_syndrome_observation")
    assert cursor.fetchone()[0] == 999


def test_observation_rejects_a_pattern_that_was_never_inserted(cursor):
    lp.create_syndrome_tables(cursor)
    cursor.execute(
        """
        INSERT INTO gold_syndrome_experiment
            (experiment_id, code_family, distance, physical_fault_rate, round_count, check_count)
        VALUES ('exp-1', 'surface_code', 3, 0.0001, 4, 4)
        """
    )
    with pytest.raises(Exception, match="foreign key|violates"):
        cursor.execute(
            """
            INSERT INTO gold_syndrome_observation
                (observation_id, experiment_id, syndrome_bits, logical_error_label, quantity, source_record_id)
            VALUES ('obs-1', 'exp-1', %s, false, 1, 'rec-x')
            """,
            (bytes(16),),
        )
    cursor.connection.rollback()


def test_observation_rejects_an_experiment_that_was_never_inserted(cursor):
    lp.create_syndrome_tables(cursor)
    cursor.execute(
        "INSERT INTO gold_syndrome_pattern (syndrome_bits, triggered_check_count) VALUES (%s, %s)",
        (ROW_A["syndrome_bits"], 1),
    )
    with pytest.raises(Exception, match="foreign key|violates"):
        cursor.execute(
            """
            INSERT INTO gold_syndrome_observation
                (observation_id, experiment_id, syndrome_bits, logical_error_label, quantity, source_record_id)
            VALUES ('obs-1', 'unknown-exp', %s, false, 1, 'rec-x')
            """,
            (ROW_A["syndrome_bits"],),
        )
    cursor.connection.rollback()


def test_pattern_rejects_syndrome_bits_of_the_wrong_length(cursor):
    lp.create_syndrome_tables(cursor)
    with pytest.raises(Exception, match="check|violates"):
        cursor.execute(
            "INSERT INTO gold_syndrome_pattern (syndrome_bits, triggered_check_count) VALUES (%s, %s)",
            (bytes(15), 0),
        )
    cursor.connection.rollback()


def test_observation_rejects_non_positive_quantity(cursor):
    lp.create_syndrome_tables(cursor)
    bad_row = {**ROW_A, "quantity": 0}
    with pytest.raises(Exception, match="check|violates"):
        lp.load_syndrome_tables(cursor, _table(bad_row))
    cursor.connection.rollback()


def test_source_record_id_is_kept_after_loading(cursor):
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A))

    cursor.execute("SELECT source_record_id FROM gold_syndrome_observation")
    assert cursor.fetchone()[0] == "rec-a"


def test_two_silver_rows_mapping_to_the_same_observation_are_rejected_loudly(cursor):
    # Two rows with the same (experiment, pattern, label) but different
    # source_record_id would otherwise silently overwrite one another's
    # quantity via ON CONFLICT. This must raise instead of doing that.
    lp.create_syndrome_tables(cursor)
    colliding_row = {**ROW_A, "source_record_id": "rec-different", "quantity": 5}

    with pytest.raises(ValueError, match="observation_id"):
        lp.load_syndrome_tables(cursor, _table(ROW_A, colliding_row))


def test_failed_load_leaves_previous_data_in_place(cursor):
    connection = cursor.connection
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A))
    connection.commit()

    bad_row = {**ROW_A, "source_record_id": "rec-bad", "quantity": 0}
    with pytest.raises(Exception):
        lp.load_syndrome_tables(cursor, _table(ROW_A, bad_row))
    connection.rollback()

    assert _count(cursor, "gold_syndrome_observation") == 1
    cursor.execute("SELECT quantity FROM gold_syndrome_observation")
    assert cursor.fetchone()[0] == 100  # ROW_A's committed value, untouched by the failed rerun


def test_row_removed_from_silver_is_removed_from_gold_on_rerun(cursor):
    lp.create_syndrome_tables(cursor)
    other_row = {
        **ROW_A,
        "source_record_id": "rec-c",
        "experiment_id": "exp-2",
        "physical_fault_rate": 0.0005,
        "syndrome_bits": DISTINCT_PATTERN_BITS,
    }
    lp.load_syndrome_tables(cursor, _table(ROW_A, other_row))
    assert _count(cursor, "gold_syndrome_experiment") == 2
    assert _count(cursor, "gold_syndrome_pattern") == 2
    assert _count(cursor, "gold_syndrome_observation") == 2

    lp.load_syndrome_tables(cursor, _table(ROW_A))  # exp-2's row no longer present in Silver

    assert _count(cursor, "gold_syndrome_experiment") == 1
    assert _count(cursor, "gold_syndrome_pattern") == 1
    assert _count(cursor, "gold_syndrome_observation") == 1
    cursor.execute("SELECT experiment_id FROM gold_syndrome_experiment")
    assert cursor.fetchone()[0] == "exp-1"


def test_experiment_carries_shared_vocabulary_columns(cursor):
    # distance comes from Silver (the d-3_ filename), code_family is the
    # source-level constant - the same two columns every source's Gold carries
    lp.create_syndrome_tables(cursor)
    lp.load_syndrome_tables(cursor, _table(ROW_A))

    cursor.execute("SELECT code_family, distance FROM gold_syndrome_experiment")
    assert cursor.fetchone() == ("surface_code", 3)


def test_experiment_rejects_unknown_code_family(cursor):
    lp.create_syndrome_tables(cursor)
    with pytest.raises(Exception, match="check|violates"):
        cursor.execute(
            """
            INSERT INTO gold_syndrome_experiment
                (experiment_id, code_family, distance, physical_fault_rate, round_count, check_count)
            VALUES ('exp-1', 'color_code', 3, 0.0001, 4, 4)
            """
        )
    cursor.connection.rollback()
