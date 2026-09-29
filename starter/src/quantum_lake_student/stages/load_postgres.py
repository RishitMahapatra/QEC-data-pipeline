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
from quantum_lake_student.models import RESULTS_PART1, StageResult, code_revision, merge_json_file
from quantum_lake_student.stages.prepare_data import SILVER_SYNDROMES_PATH

SYNDROMES_SOURCE_NAME = "qec_syndromes"

# syndrome_experiment table with one row per fault-rate file,
# with checks that the rate is a probability (between 0 and 1) and rounds and checks both equal 4
SYNDROME_EXPERIMENT_DDL = """
CREATE TABLE IF NOT EXISTS syndrome_experiment (
    experiment_id       TEXT PRIMARY KEY,
    physical_fault_rate NUMERIC NOT NULL CHECK (physical_fault_rate > 0 AND physical_fault_rate < 1),
    round_count         INT NOT NULL CHECK (round_count = 4),
    check_count         INT NOT NULL CHECK (check_count = 4)
);
COMMENT ON TABLE syndrome_experiment IS
    'One row represents one simulated physical-fault-rate experiment (one source CSV file).';
"""

# syndrome_pattern: one row per distinct 16-bit pattern, plus how many of its
# 16 checks actually fired. The CHECK is a DB-level guarantee that a pattern is
# really 16 bytes and each byte is 0 or 1 (hex "00" or "01"), on top of what
# Silver already validated.
SYNDROME_PATTERN_DDL = """
CREATE TABLE IF NOT EXISTS syndrome_pattern (
    syndrome_bits          BYTEA PRIMARY KEY CHECK (encode(syndrome_bits, 'hex') ~ '^(0[01]){16}$'),
    triggered_check_count  INT NOT NULL CHECK (triggered_check_count BETWEEN 0 AND 16)
);
COMMENT ON TABLE syndrome_pattern IS
    'One row represents one distinct 16-bit syndrome pattern, seen in any experiment.';
"""

# main table, one row per (experiment, pattern, label) combination, points at both
# parent tables via a foreign key, stores the quantity and label, and source_record_id for tracing
# indexes added for quicker queries. UNIQUE(source_record_id) guarantees every
# Gold row traces back to exactly one Silver row, never shared.
SYNDROME_OBSERVATION_DDL = """
CREATE TABLE IF NOT EXISTS syndrome_observation (
    observation_id       TEXT PRIMARY KEY,
    experiment_id        TEXT NOT NULL REFERENCES syndrome_experiment (experiment_id),
    syndrome_bits         BYTEA NOT NULL REFERENCES syndrome_pattern (syndrome_bits),
    logical_error_label   BOOLEAN NOT NULL,
    quantity              BIGINT NOT NULL CHECK (quantity > 0),
    source_record_id      TEXT NOT NULL UNIQUE,
    UNIQUE (experiment_id, syndrome_bits, logical_error_label)
);
COMMENT ON TABLE syndrome_observation IS
    'One row represents one (experiment, syndrome pattern, logical-error label) combination and its quantity weight.';
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


def _check_no_silent_collisions(rows: list[dict]) -> None:
    # raise an error if two silver rows hash to the same observation_id 
    # but have different source_record_id
    seen: dict[str, str] = {}
    for r in rows:
        observation_id = _syndrome_observation_id(
            r["experiment_id"], r["syndrome_bits"], r["logical_error_label"]
        )
        existing_source = seen.get(observation_id)
        if existing_source is not None and existing_source != r["source_record_id"]:
            raise ValueError(
                f"Silver rows {existing_source!r} and {r['source_record_id']!r} both map to "
                f"observation_id {observation_id!r}"
            )
        seen[observation_id] = r["source_record_id"]


def load_syndrome_tables(cursor, silver_table) -> None:
    # loads data from silver(parquet) into gold(postgres)
    rows = silver_table.to_pylist()
    _check_no_silent_collisions(rows)

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

    # rows that used to be in Silver but no longer shouldn't linger in Gold
    current_source_ids = [r["source_record_id"] for r in rows]
    cursor.execute(
        "DELETE FROM syndrome_observation WHERE NOT (source_record_id = ANY(%s))",
        (current_source_ids,),
    )
    current_experiment_ids = list(experiments)
    cursor.execute(
        "DELETE FROM syndrome_experiment WHERE NOT (experiment_id = ANY(%s))",
        (current_experiment_ids,),
    )
    current_patterns = list({r["syndrome_bits"] for r in rows})
    cursor.execute(
        "DELETE FROM syndrome_pattern WHERE NOT (syndrome_bits = ANY(%s))",
        (current_patterns,),
    )


def _read_silver_bytes(settings: Settings, object_name: str) -> bytes:
    # downloads the raw file bytes and returns them
    client = minio_client(settings)
    response = client.get_object(settings.s3_bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def run(run_id: str) -> StageResult:
    """Only qec_syndromes is implemented so far."""
    settings = Settings.from_environment()
    result = StageResult(stage="load_postgres", run_id=run_id)

    silver_bytes = _read_silver_bytes(settings, SILVER_SYNDROMES_PATH)
    input_hash = hashlib.sha256(silver_bytes).hexdigest()
    syndromes_table = pq.read_table(io.BytesIO(silver_bytes))

    silver_rows = syndromes_table.to_pylist()
    silver_quantity_by_experiment: dict[str, int] = {}
    for r in silver_rows:
        silver_quantity_by_experiment[r["experiment_id"]] = (
            silver_quantity_by_experiment.get(r["experiment_id"], 0) + r["quantity"]
        )

    with postgres_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS gold")
            cursor.execute("SET search_path TO gold")
            create_syndrome_tables(cursor)
            load_syndrome_tables(cursor, syndromes_table)

            cursor.execute("SELECT COUNT(*) FROM syndrome_experiment")
            experiment_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM syndrome_pattern")
            pattern_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM syndrome_observation")
            observation_count = cursor.fetchone()[0]
            cursor.execute("SELECT COALESCE(SUM(quantity), 0) FROM syndrome_observation")
            # SUM(quantity) over a BIGINT column comes back as NUMERIC (Decimal
            # in Python), not int -- json.dumps can't serialize Decimal.
            gold_quantity_total = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT experiment_id, COALESCE(SUM(quantity), 0) FROM syndrome_observation GROUP BY experiment_id"
            )
            gold_quantity_by_experiment = {
                experiment_id: int(total) for experiment_id, total in cursor.fetchall()
            }

            # every Silver row (and its quantity weight) must arrive in Gold exactly
            # once. Raising here, before the commit, rolls the whole load back, so
            # a load that lost or duplicated rows never replaces the previous Gold data
            rows_match = observation_count == syndromes_table.num_rows
            quantity_match = gold_quantity_by_experiment == silver_quantity_by_experiment
            if not (rows_match and quantity_match):
                raise ValueError(
                    f"Silver-to-Gold mismatch: {syndromes_table.num_rows} Silver rows vs "
                    f"{observation_count} Gold rows; quantity per experiment "
                    f"Silver={silver_quantity_by_experiment} Gold={gold_quantity_by_experiment}"
                )

    result.output_count = observation_count
    result.finish()

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "load_postgres",
        SYNDROMES_SOURCE_NAME,
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hash": input_hash,
            "output_row_counts": {
                "syndrome_experiment": experiment_count,
                "syndrome_pattern": pattern_count,
                "syndrome_observation": observation_count,
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "load_postgres",
        SYNDROMES_SOURCE_NAME,
        payload={
            "silver_rows": syndromes_table.num_rows,
            "gold_observation_rows": observation_count,
            "rows_match": rows_match,
            "silver_quantity_total": sum(silver_quantity_by_experiment.values()),
            "gold_quantity_total": gold_quantity_total,
            "quantity_match": quantity_match,
            "quantity_by_experiment": {
                experiment_id: {
                    "silver": silver_quantity_by_experiment.get(experiment_id, 0),
                    "gold": gold_quantity_by_experiment.get(experiment_id, 0),
                    "match": silver_quantity_by_experiment.get(experiment_id, 0)
                    == gold_quantity_by_experiment.get(experiment_id, 0),
                }
                for experiment_id in silver_quantity_by_experiment
            },
        },
    )

    return result
