"""Tests for the Gold load_postgres stage (google_qec source).

Runs against the real Postgres instance, but entirely inside a throwaway
schema (test_gold_<random>) so these tests never touch the real gold_*
tables. The schema is created before each test and dropped afterward via a
fixture, regardless of pass/fail. No MinIO/Bronze/Silver is touched - the
Silver tables these tests load from are built in-memory as small PyArrow
tables, not read from the real lake.
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

    yield settings, schema_name, conn

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema_name} CASCADE;")
    conn.close()


ONE_EXPERIMENT = [{
    "source_record_id": "x", "experiment_id": "exp-1", "basis": "X",
    "distance": 3, "rounds": 25, "shots": 2, "center_row": 3, "center_col": 5,
    "measurement_count": 10, "detector_count": 8,
}]
TWO_SHOTS = [
    {
        "source_record_id": "s0", "experiment_id": "exp-1", "shot_index": 0,
        "measurement_bits": b"\x00\x00", "sweep_bits": b"\x00\x00", "detector_bits": b"\x00",
        "detector_event_count": 0, "actual_observable_flip": False,
        "belief_matching_prediction": False, "correlated_matching_prediction": False,
        "pymatching_prediction": False, "tensor_network_contraction_prediction": False,
    },
    {
        "source_record_id": "s1", "experiment_id": "exp-1", "shot_index": 1,
        "measurement_bits": b"\xff\x01", "sweep_bits": b"\x01\x00", "detector_bits": b"\x01",
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


def test_schema_creates_experiment_and_shot_tables(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
            (schema_name,),
        )
        tables = {row[0] for row in cur.fetchall()}
    assert {"gold_google_experiment", "gold_google_shot"} <= tables


def test_shot_foreign_key_rejects_unknown_experiment(pg_schema) -> None:
    settings, schema_name, conn = pg_schema
    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                """
                INSERT INTO gold_google_shot
                    (experiment_id, shot_index, measurement_bits, sweep_bits, detector_bits,
                     detector_event_count, actual_observable_flip,
                     belief_matching_prediction, correlated_matching_prediction,
                     pymatching_prediction, tensor_network_contraction_prediction)
                VALUES ('does-not-exist', 0, %s, %s, %s, 0, false, false, false, false, false)
                """,
                (b"\x00", b"\x00", b"\x00"),
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
                    (experiment_id, basis, distance, rounds, shots, center_row, center_col,
                     measurement_count, detector_count)
                VALUES ('exp-bad', 'X', 3, 25, 0, 3, 5, 10, 8)
                """
            )
    conn.rollback()


def test_load_google_gold_inserts_expected_row_counts(pg_schema, monkeypatch) -> None:
    settings, schema_name, conn = pg_schema
    monkeypatch.setattr(lp, "_read_silver_parquet", _fake_read)

    with conn.cursor() as cur:
        cur.execute(lp.GOOGLE_GOLD_SCHEMA_SQL)

    exp_count, shot_count = lp.load_google_gold(conn, settings)
    conn.commit()

    assert (exp_count, shot_count) == (1, 2)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_google_shot;")
        assert cur.fetchone()[0] == 2


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


def test_run_writes_gold_and_run_evidence(pg_schema, monkeypatch, tmp_path) -> None:
    """Also runs `run()` against the throwaway schema. run() calls
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