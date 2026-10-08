"""
This file creates the required analyses and ML input tables from PostgreSQL Gold.

In this stage we write the Gold queries/views, joins, stable example IDs, and export logic.
A thin export step may execute SQL, use the course split helpers, validate the
fixed contracts, and write Parquet. It does not re-read Bronze or Silver .

All data logic (joins, example_id, the decoder pivot) lives in the Gold views v_ml_syndrome_decoder_example and
v_ml_google_decoder_example, created by load_postgres.run(). 
Here we only:

1) select from those views,
2) add data_split with the course helpers (syndrome_data_split, google_data_split),
3) check both contracts from assignment/required-ml-tables.md,
4) check every example_id resolves to its Gold record(s),
5) write ml/ml_syndrome_decoder_example.parquet and ml/ml_google_decoder_example.parquet,
6) run the three Part I analysis queries and save their results next to the SQL.

"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student import connections
from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    MODEL_SPLITS,
    google_data_split,
    google_meta_model_input,
    syndrome_data_split,
    syndrome_model_input,
)
from quantum_lake_student.models import RESULTS_PART1, StageResult, code_revision, merge_json_file
from quantum_lake_student.stages import load_postgres, register_sources

SYNDROME_VIEW = "v_ml_syndrome_decoder_example"
GOOGLE_VIEW = "v_ml_google_decoder_example"
SYNDROME_ML_PATH = "ml/ml_syndrome_decoder_example.parquet"
GOOGLE_ML_PATH = "ml/ml_google_decoder_example.parquet"

# Exact column names, order and types from assignment/required-ml-tables.md.
# nullable=False: every value is required; _check_no_nulls enforces it.
SYNDROME_ML_SCHEMA = pa.schema([
    pa.field("example_id", pa.string(), nullable=False),
    pa.field("experiment_id", pa.string(), nullable=False),
    pa.field("physical_fault_rate", pa.float64(), nullable=False),
    pa.field("syndrome_bits", pa.binary(), nullable=False),
    pa.field("round_count", pa.int32(), nullable=False),
    pa.field("check_count", pa.int32(), nullable=False),
    pa.field("logical_error_label", pa.bool_(), nullable=False),
    pa.field("sample_weight", pa.int64(), nullable=False),
    pa.field("data_split", pa.string(), nullable=False),
])

GOOGLE_ML_SCHEMA = pa.schema([
    pa.field("example_id", pa.string(), nullable=False),
    pa.field("experiment_id", pa.string(), nullable=False),
    pa.field("shot_index", pa.int64(), nullable=False),
    pa.field("distance", pa.int32(), nullable=False),
    pa.field("rounds", pa.int32(), nullable=False),
    pa.field("center_row", pa.int32(), nullable=False),
    pa.field("center_col", pa.int32(), nullable=False),
    pa.field("detector_count", pa.int32(), nullable=False),
    pa.field("detector_event_count", pa.int32(), nullable=False),
    pa.field("detector_bits", pa.binary(), nullable=False),
    *(pa.field(name, pa.bool_(), nullable=False) for name in GOOGLE_META_PREDICTION_COLUMNS),
    pa.field("actual_observable_flip", pa.bool_(), nullable=False),
    pa.field("data_split", pa.string(), nullable=False),
])


class MLContractError(ValueError):
    """An ML table does not satisfy its required-ml-tables.md contract."""


# ---------------------------------------------------------------------------
# Export: select from the Gold views, add data_split
# ---------------------------------------------------------------------------


def _select_view(conn, view: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {view}")
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def export_syndrome_table(conn) -> pa.Table:
    rows = _select_view(conn, SYNDROME_VIEW)
    for r in rows:
        r["data_split"] = syndrome_data_split(r["physical_fault_rate"])
    return pa.Table.from_pylist(rows, schema=SYNDROME_ML_SCHEMA)


def export_google_table(conn) -> pa.Table:
    rows = _select_view(conn, GOOGLE_VIEW)
    for r in rows:
        r["data_split"] = google_data_split(r["shot_index"])
    return pa.Table.from_pylist(rows, schema=GOOGLE_ML_SCHEMA)


# ---------------------------------------------------------------------------
# Contract checks (required-ml-tables.md, "Checks your code must perform")
# ---------------------------------------------------------------------------


def _check_no_nulls(table: pa.Table, errors: list[str]) -> None:
    for name in table.column_names:
        if table[name].null_count:
            errors.append(f"{name} has {table[name].null_count} null values")


def _check_ids_and_splits(table: pa.Table, errors: list[str]) -> None:
    ids = table["example_id"].to_pylist()
    if len(set(ids)) != len(ids):
        errors.append(f"example_id is not unique ({len(ids) - len(set(ids))} duplicates)")
    present = set(table["data_split"].to_pylist())
    for split in MODEL_SPLITS:
        if split not in present:
            errors.append(f"split {split!r} is empty")


def _raise_if_errors(table_name: str, errors: list[str]) -> None:
    if errors:
        shown = "; ".join(errors[:5]) + (f"; ... {len(errors) - 5} more" if len(errors) > 5 else "")
        raise MLContractError(f"{table_name}: {shown}")


def check_syndrome_contract(table: pa.Table) -> None:
    errors: list[str] = []
    _check_no_nulls(table, errors)
    _check_ids_and_splits(table, errors)
    for r in table.to_pylist():
        try:
            syndrome_model_input(r["syndrome_bits"])  # exactly 16 binary values
        except (TypeError, ValueError) as exc:
            errors.append(f"{r['example_id']}: {exc}")
        if r["round_count"] != 4 or r["check_count"] != 4:
            errors.append(f"{r['example_id']}: round_count/check_count must be 4/4")
        if r["sample_weight"] is None or r["sample_weight"] <= 0:
            errors.append(f"{r['example_id']}: sample_weight must be positive")
    _raise_if_errors("ml_syndrome_decoder_example", errors)


def check_google_contract(table: pa.Table) -> None:
    errors: list[str] = []
    _check_no_nulls(table, errors)
    _check_ids_and_splits(table, errors)

    # distance-3 and distance-5 rows must each have one feature width, so Part II can build their raw-detector matrices separately
    widths: dict[int, set[int]] = {}
    for distance, count in zip(table["distance"].to_pylist(), table["detector_count"].to_pylist()):
        widths.setdefault(distance, set()).add(count)
    for distance, counts in widths.items():
        if len(counts) != 1:
            errors.append(f"distance {distance} has several detector_count values {sorted(counts)}")

    for r in table.to_pylist():
        bits, count = r["detector_bits"], r["detector_count"]
        if bits is None or count is None:
            continue  # already reported by _check_no_nulls
        if len(bits) != (count + 7) // 8:
            errors.append(f"{r['example_id']}: {len(bits)} detector bytes, expected ceil({count}/8)")
            continue
        # bits are little-endian inside each byte, so unused padding bits are the high bits of the last byte
        if count % 8 and bits[-1] >> (count % 8):
            errors.append(f"{r['example_id']}: unused padding bits are not zero")
        if int.from_bytes(bits, "little").bit_count() != r["detector_event_count"]:
            errors.append(f"{r['example_id']}: detector_event_count does not equal the number of set bits")
        try:
            google_meta_model_input(r)  # event density + four binary predictions
        except (TypeError, ValueError) as exc:
            errors.append(f"{r['example_id']}: {exc}")
    _raise_if_errors("ml_google_decoder_example", errors)


# ---------------------------------------------------------------------------
# Tracing: every example_id resolves to its Gold record(s)
# ---------------------------------------------------------------------------

RESOLUTION_SQL = {
    # example_id is gold_syndrome_observation's primary key
    "ml_syndrome_decoder_example": (
        "SELECT COUNT(*) FROM gold_syndrome_observation",
        f"SELECT COUNT(*) FROM {SYNDROME_VIEW} v "
        "LEFT JOIN gold_syndrome_observation o ON o.observation_id = v.example_id "
        "WHERE o.observation_id IS NULL",
    ),
    # example_id resolves through the view to gold_google_shot's primary key
    "ml_google_decoder_example": (
        "SELECT COUNT(*) FROM gold_google_shot",
        f"SELECT COUNT(*) FROM {GOOGLE_VIEW} v "
        "LEFT JOIN gold_google_shot s ON s.experiment_id = v.experiment_id AND s.shot_index = v.shot_index "
        "WHERE s.experiment_id IS NULL",
    ),
}


def gold_resolution_counts(conn) -> dict[str, dict[str, int]]:
    """Gold row count and number of unresolved example_ids per ML table."""
    counts = {}
    with conn.cursor() as cur:
        for table_name, (gold_sql, unresolved_sql) in RESOLUTION_SQL.items():
            cur.execute(gold_sql)
            gold_rows = cur.fetchone()[0]
            cur.execute(unresolved_sql)
            counts[table_name] = {"gold_rows": gold_rows, "unresolved_example_ids": cur.fetchone()[0]}
    return counts


# ---------------------------------------------------------------------------
# Part I analyses: run each committed query*.sql against Gold, save its result
# ---------------------------------------------------------------------------

# The SQL files start with `SET search_path TO gold;` so they also run ad hoc
# in psql. The stage has already set the search_path to the schema it was
# given (tests use a throwaway schema), so that line is skipped here.
_SET_GOLD_SEARCH_PATH = re.compile(r"^\s*SET\s+search_path\s+TO\s+gold\s*;", re.IGNORECASE | re.MULTILINE)


def _csv_value(value):
    if isinstance(value, (bytes, memoryview)):
        return bytes(value).hex()
    if isinstance(value, list):
        return "{" + ",".join(str(v) for v in value) + "}"
    return value


def run_analyses(conn, analysis_dir: Path) -> dict[str, int]:
    """Run every query*.sql in analysis_dir and write <name>_result.csv next to it.

    Returns {result file name: row count}.
    """
    outputs = {}
    for sql_path in sorted(analysis_dir.glob("query*.sql")):
        sql = _SET_GOLD_SEARCH_PATH.sub("", sql_path.read_text())
        with conn.cursor() as cur:
            cur.execute(sql)
            columns, rows = None, []
            while True:  # the answer is the last statement that returns rows
                if cur.description is not None:
                    columns, rows = [d.name for d in cur.description], cur.fetchall()
                if not cur.nextset():
                    break
        if columns is None:
            raise RuntimeError(f"{sql_path.name} returned no result set")
        result_path = sql_path.with_name(f"{sql_path.stem}_result.csv")
        with result_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows([_csv_value(v) for v in row] for row in rows)
        outputs[result_path.name] = len(rows)
    return outputs


# ---------------------------------------------------------------------------
# Writing and run()
# ---------------------------------------------------------------------------


def _parquet_bytes(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def _write_lake_object(settings: Settings, object_name: str, data: bytes) -> None:
    """Write one ML Parquet file to the lake (MinIO bucket or local lake folder)."""
    if settings.lake_backend == "local":
        path = settings.local_lake_root / object_name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)  # never leave a half-written file behind
        return
    minio_client(settings).put_object(
        settings.s3_bucket,
        object_name,
        io.BytesIO(data),
        len(data),
        content_type="application/vnd.apache.parquet",
    )


def _data_release() -> str:
    """Course data-release identifier from the supplied bundle manifest."""
    manifest = register_sources._load_manifest()
    return f"{manifest['release_name']} v{manifest['bundle_version']} ({manifest['release_date']})"


def _split_counts(table: pa.Table) -> dict[str, int]:
    splits = table["data_split"].to_pylist()
    return {split: splits.count(split) for split in MODEL_SPLITS}


def run(run_id: str) -> StageResult:
    """Export both ML tables and the Part I analysis results from Gold."""
    settings = Settings.from_environment()
    result = StageResult(stage="build_ml_tables", run_id=run_id)
    analysis_dir = RESULTS_PART1 / "analysis"

    conn = connections.postgres_connection(settings)
    # one REPEATABLE READ snapshot: both tables, the resolution checks and the
    # analyses all see the same Gold, even if a load commits meanwhile
    conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {load_postgres.GOLD_SCHEMA}")
        syndrome_table = export_syndrome_table(conn)
        google_table = export_google_table(conn)
        resolution = gold_resolution_counts(conn)
        analysis_outputs = run_analyses(conn, analysis_dir)
    finally:
        conn.rollback()  # read-only stage: nothing to commit
        conn.close()

    check_syndrome_contract(syndrome_table)
    check_google_contract(google_table)

    ml_rows = {
        "ml_syndrome_decoder_example": syndrome_table.num_rows,
        "ml_google_decoder_example": google_table.num_rows,
    }
    for table_name, counts in resolution.items():
        if counts["unresolved_example_ids"] or counts["gold_rows"] != ml_rows[table_name]:
            raise RuntimeError(
                f"{table_name}: {ml_rows[table_name]} ML rows vs {counts['gold_rows']} Gold rows, "
                f"{counts['unresolved_example_ids']} example_ids do not resolve to Gold"
            )

    # everything is checked before anything is written
    syndrome_bytes = _parquet_bytes(syndrome_table)
    google_bytes = _parquet_bytes(google_table)
    _write_lake_object(settings, SYNDROME_ML_PATH, syndrome_bytes)
    _write_lake_object(settings, GOOGLE_ML_PATH, google_bytes)

    result.input_count = sum(counts["gold_rows"] for counts in resolution.values())
    result.output_count = sum(ml_rows.values())
    result.issue_count = 0
    result.finish()

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "build_ml_tables",
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "data_release": _data_release(),
            "input_views": [SYNDROME_VIEW, GOOGLE_VIEW],
            "output_hashes": {
                SYNDROME_ML_PATH: hashlib.sha256(syndrome_bytes).hexdigest(),
                GOOGLE_ML_PATH: hashlib.sha256(google_bytes).hexdigest(),
            },
            "output_row_counts": {
                SYNDROME_ML_PATH: syndrome_table.num_rows,
                GOOGLE_ML_PATH: google_table.num_rows,
                **{f"results/part1/analysis/{name}": rows for name, rows in analysis_outputs.items()},
            },
            "split_counts": {
                SYNDROME_ML_PATH: _split_counts(syndrome_table),
                GOOGLE_ML_PATH: _split_counts(google_table),
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "build_ml_tables",
        payload={
            table_name: {
                "gold_rows": counts["gold_rows"],
                "ml_rows": ml_rows[table_name],
                "unresolved_example_ids": counts["unresolved_example_ids"],
                "match": counts["gold_rows"] == ml_rows[table_name] and counts["unresolved_example_ids"] == 0,
            }
            for table_name, counts in resolution.items()
        },
    )
    return result
