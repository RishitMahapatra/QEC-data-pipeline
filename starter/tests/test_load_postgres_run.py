"""Tests for the combined Gold run(): calls both sources, sums their counts.

google_qec's and qasmbench's own loaders are covered in depth in
test_load_postgres_google.py and test_gold_qasmbench.py (schema, foreign
keys, value checks, idempotency, all-or-nothing rollback). This file only
checks the umbrella run() wires the two together correctly, the same
relationship prepare_data.py's Silver run() has to each source's own
builder.
"""

from __future__ import annotations

import uuid

import psycopg
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
    """Same throwaway-schema fixture as the per-source test files, shared
    by both run_google_gold() and run_qasmbench_gold() inside run()."""
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


@needs_postgres
def test_run_calls_both_sources_and_sums_counts(pg_schema, monkeypatch, tmp_path) -> None:
    settings, schema_name, _fixture_conn = pg_schema

    google_result = lp.StageResult(stage="load_postgres:google_qec", run_id="combined-run")
    google_result.input_count, google_result.output_count, google_result.issue_count = 5, 7, 0
    qasmbench_result = lp.StageResult(stage="load_postgres:qasmbench", run_id="combined-run")
    qasmbench_result.input_count, qasmbench_result.output_count, qasmbench_result.issue_count = 3, 4, 0

    calls: list[str] = []

    def _fake_run_google_gold(_settings, _run_id):
        calls.append("google_qec")
        return google_result

    def _fake_run_qasmbench_gold(_settings, _run_id):
        calls.append("qasmbench")
        return qasmbench_result

    monkeypatch.setattr(lp, "run_google_gold", _fake_run_google_gold)
    monkeypatch.setattr(lp, "run_qasmbench_gold", _fake_run_qasmbench_gold)

    def _test_postgres_connection(_settings):
        c = psycopg.connect(settings.postgres_dsn)
        with c.cursor() as cur:
            cur.execute(f"SET search_path TO {schema_name};")
        return c

    monkeypatch.setattr(lp.connections, "postgres_connection", _test_postgres_connection)
    monkeypatch.setenv("LAKE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LAKE_ROOT", str(tmp_path))

    result = lp.run("combined-run")

    assert calls == ["google_qec", "qasmbench"]  # google before qasmbench, both called exactly once
    assert result.stage == "load_postgres"
    assert result.input_count == 5 + 3
    assert result.output_count == 7 + 4
    assert result.finished_at is not None


@needs_postgres
def test_run_end_to_end_loads_both_sources_for_real(pg_schema, monkeypatch, tmp_path) -> None:
    """A lighter-weight but real integration check: no monkeypatching of the
    per-source functions themselves, only of the connection and the Silver
    reader, so run() actually executes both sources' schema DDL and inserts
    against the throwaway schema."""
    settings, schema_name, _fixture_conn = pg_schema

    from test_gold_qasmbench import _fake_read as qasmbench_fake_read
    from test_load_postgres_google import _fake_read as google_fake_read

    def _combined_fake_read(_settings, object_name: str):
        if "google_qec" in object_name:
            return google_fake_read(_settings, object_name)
        return qasmbench_fake_read(_settings, object_name)

    monkeypatch.setattr(lp, "_read_silver_parquet", _combined_fake_read)

    def _test_postgres_connection(_settings):
        c = psycopg.connect(settings.postgres_dsn)
        with c.cursor() as cur:
            cur.execute(f"SET search_path TO {schema_name};")
        return c

    monkeypatch.setattr(lp.connections, "postgres_connection", _test_postgres_connection)
    results_root = tmp_path / "results" / "part1"
    monkeypatch.setattr(lp, "RESULTS_PART1", results_root)

    result = lp.run("combined-run-real")

    assert result.stage == "load_postgres"
    assert result.output_count == 3 + 10  # google_qec: 1 exp + 2 shots; qasmbench: 1+4+2+3
    assert (results_root / "run.json").exists()
    assert (results_root / "row_counts.json").exists()

    with _fixture_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_google_experiment;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        assert cur.fetchone()[0] == 1
