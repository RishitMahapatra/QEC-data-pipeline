"""Tests for the Gold load_postgres stage (google_qec source).

Runs against the real Postgres instance, but entirely inside a throwaway
schema (test_gold_<random>) so these tests never touch the real gold_*
tables. The schema is created before each test and dropped afterward via a
fixture, regardless of pass/fail. No MinIO/Bronze/Silver is touched - the
Silver tables these tests load from are built in-memory as small PyArrow
tables, not read from the real lake.

Covers: table creation, source_record_id tracing, the decoder/decoder_
prediction tables, all four data-driven constraints (distance<->detector_
count, basis in X/Z, unique processor location, packed detector-byte
length), row-count reconciliation, and the all-or-nothing rollback
guarantee.
"""

from __future__ import annotations

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
    real 'public' gold_google_* tables."""
    settings = Settings.from_environment()
    schema_name = f"test_gold_{uuid.uuid4().hex[:8]}"

    conn = psycopg.connect(settings.postgres_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema_name};")
        cur.execute(f"SET search_path TO {schema_name};")
    conn.autocommit = False  # <-- ADD THIS: tests need real transactions to test rollback

    yield settings, schema_name, conn

    conn.rollback()  # <-- ADD THIS: clean up any uncommitted state before dropping
    conn.autocommit = True  # <-- ADD THIS: DROP SCHEMA needs autocommit again
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
        "measurement_bits": b"\x00\x00", "sweep_bits": b"\x00\x00", "detector_bits": b"\x00" * 25,
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
# Foreign keys and basic constraints
# ---------------------------------------------------------------------------


def test_shot_foreign_key_rejects_unknown_experiment(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, source_record_id, shot_index, measurement_bits, sweep_bits, detector_bits,
                     detector_event_count, actual_observable_flip)
                VALUES ('does-not-exist', 'x', 0, %s, %s, %s, 0, false)
                """,
                (b"\x00\x00", b"\x00\x00", b"\x00" * 25),
            )
    conn.rollback()


def test_experiment_rejects_nonpositive_shots(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO gold_google_experiment
                    (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col,
                     measurement_count, detector_count)
                VALUES ('exp-bad', 'x', 'X', 3, 25, 0, 3, 5, 10, 200)
                """
            )
    conn.rollback()


# ---------------------------------------------------------------------------
# The four data-driven constraints from Martina's review
# ---------------------------------------------------------------------------


def test_experiment_rejects_detector_count_not_matching_distance(pg_schema) -> None:
    """distance=3 must have detector_count=200; a mismatched pair (e.g.
    distance=3 with detector_count=600) must be rejected."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO gold_google_experiment
                    (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col,
                     measurement_count, detector_count)
                VALUES ('exp-mismatch', 'x', 'X', 3, 25, 100, 3, 5, 10, 600)
                """
            )
    conn.rollback()


def test_experiment_accepts_both_valid_distance_detector_pairs(pg_schema) -> None:
    """distance=3/detector_count=200 and distance=5/detector_count=600 must
    both be accepted - this is the valid side of the same constraint."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        cur.execute(
            """
            INSERT INTO gold_google_experiment
                (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col,
                 measurement_count, detector_count)
            VALUES ('exp-d3', 'x', 'X', 3, 25, 100, 3, 5, 10, 200)
            """
        )
        cur.execute(
            """
            INSERT INTO gold_google_experiment
                (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col,
                 measurement_count, detector_count)
            VALUES ('exp-d5', 'x', 'X', 5, 25, 100, 9, 9, 10, 600)
            """
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 2


def test_experiment_rejects_basis_outside_x_or_z(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO gold_google_experiment
                    (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col,
                     measurement_count, detector_count)
                VALUES ('exp-bad-basis', 'x', 'Y', 3, 25, 100, 3, 5, 10, 200)
                """
            )
    conn.rollback()


def test_experiment_rejects_duplicate_processor_location_and_settings(pg_schema) -> None:
    """Two experiments can't share the same (distance, rounds, center_row,
    center_col) - that combination must be unique."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        cur.execute(
            """
            INSERT INTO gold_google_experiment
                (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col,
                 measurement_count, detector_count)
            VALUES ('exp-first', 'x', 'X', 3, 25, 100, 3, 5, 10, 200)
            """
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                """
                INSERT INTO gold_google_experiment
                    (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col,
                     measurement_count, detector_count)
                VALUES ('exp-duplicate-location', 'x', 'X', 3, 25, 100, 3, 5, 10, 200)
                """
            )
    conn.rollback()


def test_shot_rejects_detector_bits_with_wrong_byte_length(pg_schema) -> None:
    """detector_bits must be exactly 25 or 75 bytes - anything else must be rejected."""
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        cur.execute(
            """
            INSERT INTO gold_google_experiment
                (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col,
                 measurement_count, detector_count)
            VALUES ('exp-1', 'x', 'X', 3, 25, 100, 3, 5, 10, 200)
            """
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, source_record_id, shot_index, measurement_bits, sweep_bits, detector_bits,
                     detector_event_count, actual_observable_flip)
                VALUES ('exp-1', 'x', 0, %s, %s, %s, 0, false)
                """,
                (b"\x00\x00", b"\x00\x00", b"\x00" * 10),  # wrong length - neither 25 nor 75
            )
    conn.rollback()


# ---------------------------------------------------------------------------
# load_google_gold: source_record_id tracing, decoder tables, counts
# ---------------------------------------------------------------------------


def test_load_google_gold_inserts_expected_row_counts(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)

    exp_count, shot_count, decoder_prediction_count = lp.load_google_gold(conn, settings)
    conn.commit()

    assert (exp_count, shot_count, decoder_prediction_count) == (1, 2, 8)  # 2 shots x 4 decoders
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
    """Every Gold experiment/shot row must carry the same source_record_id
    that was on its Silver row, so a Gold row can be traced back."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
    lp.load_google_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT source_record_id FROM gold_google_experiment WHERE experiment_id = 'exp-1';")
        assert cur.fetchone()[0] == "exp-src-1"

        cur.execute(
            "SELECT shot_index, source_record_id FROM gold_google_shot ORDER BY shot_index;"
        )
        rows = cur.fetchall()
        assert rows == [(0, "shot-src-0"), (1, "shot-src-1")]


def test_load_google_gold_decoder_prediction_marks_wrong_correctly(pg_schema, monkeypatch) -> None:
    """was_wrong must be true exactly when predicted_flip != actual_observable_flip."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
    lp.load_google_gold(conn, settings)
    conn.commit()

    with conn.cursor() as cur:
        # shot 0: actual_observable_flip = False, all four predictions False -> none wrong
        cur.execute(
            "SELECT decoder_name, predicted_flip, was_wrong FROM gold_google_decoder_prediction "
            "WHERE experiment_id = 'exp-1' AND shot_index = 0 ORDER BY decoder_name;"
        )
        assert all(not was_wrong for _name, _predicted, was_wrong in cur.fetchall())

        # shot 1: actual_observable_flip = True; correlated_matching predicted False -> wrong
        cur.execute(
            "SELECT was_wrong FROM gold_google_decoder_prediction "
            "WHERE experiment_id = 'exp-1' AND shot_index = 1 AND decoder_name = 'correlated_matching';"
        )
        assert cur.fetchone()[0] is True

        cur.execute(
            "SELECT was_wrong FROM gold_google_decoder_prediction "
            "WHERE experiment_id = 'exp-1' AND shot_index = 1 AND decoder_name = 'belief_matching';"
        )
        assert cur.fetchone()[0] is False


def test_load_google_gold_is_repeatable_without_duplicating(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

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
        cur.execute("SELECT COUNT(*) FROM gold_google_decoder;")
        assert cur.fetchone()[0] == 4
        cur.execute("SELECT COUNT(*) FROM gold_google_decoder_prediction;")
        assert cur.fetchone()[0] == 8


# ---------------------------------------------------------------------------
# All-or-nothing: a failed load must leave the previous complete data intact
# ---------------------------------------------------------------------------


def test_failed_load_keeps_previous_data_intact(pg_schema, monkeypatch) -> None:
    """Load once successfully, then simulate a failure partway through a
    second load (bad Silver data). The previous complete data must remain -
    nothing should be half-overwritten."""
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)

    # first, good load
    lp.load_google_gold(conn, settings)
    conn.commit()

    # second attempt: Silver data now violates a constraint (bad basis) -
    # load_google_gold should raise, and the caller is expected to roll back
    bad_experiment = [{**ONE_EXPERIMENT[0], "basis": "Y"}]

    def _fake_read_bad(_settings, object_name: str) -> pa.Table:
        return (
            pa.Table.from_pylist(bad_experiment)
            if "experiment" in object_name
            else pa.Table.from_pylist(TWO_SHOTS)
        )

    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read_bad)

    with pytest.raises(psycopg.errors.CheckViolation):
        lp.load_google_gold(conn, settings)
    conn.rollback()  # this is what run()'s except-block does on failure

    # previous good data must still be there, untouched
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT basis FROM gold_google_experiment WHERE experiment_id = 'exp-1';")
        assert cur.fetchone()[0] == "X"  # the original valid value, not overwritten
        cur.execute("SELECT COUNT(*) FROM gold_google_shot;")
        assert cur.fetchone()[0] == 2


# ---------------------------------------------------------------------------
# Full run(): schema + load + evidence files, in the isolated test schema
# ---------------------------------------------------------------------------


def test_run_writes_gold_and_run_evidence(pg_schema, monkeypatch, tmp_path) -> None:
    """Runs the full run() against the throwaway schema. run() calls
    connections.postgres_connection() internally, so we redirect it to a
    freshly-opened connection whose search_path is already the test schema."""
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

    result = lp.run("gold-test-run")

    assert (result.stage, result.output_count) == ("load_postgres", 3)
    assert (results_root / "run.json").exists()
    assert (results_root / "row_counts.json").exists()

    with _fixture_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_google_shot;")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM gold_google_decoder_prediction;")
        assert cur.fetchone()[0] == 8