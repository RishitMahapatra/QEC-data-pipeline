"""Relational integration stage.

Create the student-designed PostgreSQL tables and load them as one all-or-
nothing update. Primary/foreign keys, value checks, and indexes are part of the
deliverable. Repeated loads must not create duplicate records.
"""

import hashlib
import io

import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.models import StageResult
from quantum_lake_student.stages.prepare_data import SILVER_SYNDROMES_PATH

# syndrome_experiment table with one row per fault-rate file,
# with checks that the rate is positive and rounds and checks both equal 4
SYNDROME_EXPERIMENT_DDL = """
CREATE TABLE IF NOT EXISTS syndrome_experiment (
    experiment_id       TEXT PRIMARY KEY,
    physical_fault_rate NUMERIC NOT NULL CHECK (physical_fault_rate > 0),
    round_count         INT NOT NULL CHECK (round_count = 4),
    check_count         INT NOT NULL CHECK (check_count = 4)
);
"""

# syndrome_pattern: one row per distinct 16-bit pattern, plus how many of its
# 16 checks actually fired
SYNDROME_PATTERN_DDL = """
CREATE TABLE IF NOT EXISTS syndrome_pattern (
    syndrome_bits          BYTEA PRIMARY KEY,
    triggered_check_count  INT NOT NULL CHECK (triggered_check_count BETWEEN 0 AND 16)
);
"""

# main table, one row per (experiment, pattern, label) combination, points at both
# parent tables via a foreign key, stores the quantity and label, and source_record_id for tracing
# indexes added for quicker queries
SYNDROME_OBSERVATION_DDL = """
CREATE TABLE IF NOT EXISTS syndrome_observation (
    observation_id       TEXT PRIMARY KEY,
    experiment_id        TEXT NOT NULL REFERENCES syndrome_experiment (experiment_id),
    syndrome_bits         BYTEA NOT NULL REFERENCES syndrome_pattern (syndrome_bits),
    logical_error_label   BOOLEAN NOT NULL,
    quantity              BIGINT NOT NULL CHECK (quantity > 0),
    source_record_id      TEXT NOT NULL,
    UNIQUE (experiment_id, syndrome_bits, logical_error_label)
);
CREATE INDEX IF NOT EXISTS idx_syndrome_observation_experiment ON syndrome_observation (experiment_id);
CREATE INDEX IF NOT EXISTS idx_syndrome_observation_syndrome_bits ON syndrome_observation (syndrome_bits);
"""


def _triggered_check_count(syndrome_bits: bytes) -> int:
    # returns how many of the 16 bits are set to 1
    return sum(syndrome_bits)


def _syndrome_observation_id(experiment_id: str, syndrome_bits: bytes, label: bool) -> str:
    # hashes experiments + pattern + label into one deterministic id
    # that is easier to reference later instead of a composite key
    key = f"{experiment_id}:{syndrome_bits.hex()}:{label}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def create_syndrome_tables(cursor) -> None:
    # sends the three "create table" to postgress
    cursor.execute(SYNDROME_EXPERIMENT_DDL)
    cursor.execute(SYNDROME_PATTERN_DDL)
    cursor.execute(SYNDROME_OBSERVATION_DDL)


def load_syndrome_tables(cursor, silver_table) -> None:
    # loads data from silver(parquet) into gold(postgres)
    rows = silver_table.to_pylist()

    experiments = {
        r["experiment_id"]: (r["physical_fault_rate"], r["round_count"], r["check_count"])
        for r in rows
    }
    for experiment_id, (rate, rounds, checks) in experiments.items():
        cursor.execute(
            """
            INSERT INTO syndrome_experiment (experiment_id, physical_fault_rate, round_count, check_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (experiment_id) DO UPDATE
            SET physical_fault_rate = EXCLUDED.physical_fault_rate
            """,
            (experiment_id, rate, rounds, checks),
        )

    for bits in {r["syndrome_bits"] for r in rows}:
        cursor.execute(
            """
            INSERT INTO syndrome_pattern (syndrome_bits, triggered_check_count)
            VALUES (%s, %s)
            ON CONFLICT (syndrome_bits) DO NOTHING
            """,
            (bits, _triggered_check_count(bits)),
        )

    for r in rows:
        observation_id = _syndrome_observation_id(
            r["experiment_id"], r["syndrome_bits"], r["logical_error_label"]
        )
        cursor.execute(
            """
            INSERT INTO syndrome_observation
                (observation_id, experiment_id, syndrome_bits, logical_error_label, quantity, source_record_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (observation_id) DO UPDATE
            SET quantity = EXCLUDED.quantity, source_record_id = EXCLUDED.source_record_id
            """,
            (observation_id, r["experiment_id"], r["syndrome_bits"], r["logical_error_label"], r["quantity"], r["source_record_id"]),
        )


def _read_silver_table(settings: Settings, object_name: str):
    # fetches the parquet bytes and loads them into memory
    client = minio_client(settings)
    response = client.get_object(settings.s3_bucket, object_name)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    return pq.read_table(io.BytesIO(data))


def run(run_id: str) -> StageResult:
    """Only qec_syndromes is implemented so far
    """
    settings = Settings.from_environment()
    result = StageResult(stage="load_postgres", run_id=run_id)

    syndromes_table = _read_silver_table(settings, SILVER_SYNDROMES_PATH)

    with postgres_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS gold")
            cursor.execute("SET search_path TO gold")
            create_syndrome_tables(cursor)
            load_syndrome_tables(cursor, syndromes_table)

    result.output_count = syndromes_table.num_rows
    result.finish()
    return result
