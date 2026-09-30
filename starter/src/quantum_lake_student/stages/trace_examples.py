"""Write results/part1/trace_examples.json: one full syndrome and one full Google trace.

brief.md, "Source tracing": every ML example_id resolves to Gold, Gold rows
carry the Silver source_record_id, and source_trace.parquet maps that to the
Bronze object, archive member, record position and hash. This stage follows
one example of each ML table down that chain:

    ML row (example_id)
      -> Gold record(s)           via the v_ml_* views / primary keys
      -> Silver row               via source_record_id
      -> source_trace rows        via source_record_id
      -> raw Bronze record(s)     the archive member and record position those rows name

and checks at every hop that the values are the same (syndrome bits, label,
weight; packed detector bytes, actual flip, four decoder predictions). If
Part II has written results/part2/predictions.parquet, the prediction rows
for the traced example_id are added on top.

The examples are chosen deterministically, so reruns trace the same records:
- syndrome: the test-split example with a logical error and the largest
  sample_weight (ties broken by example_id);
- Google: the lowest test-split shot of the first distance-3 experiment.

Unlike build_ml_tables, this stage deliberately reads Silver and Bronze: its
job is to prove the path back to them. It never writes to any data zone.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quantum_lake_student import connections
from quantum_lake_student.config import Settings
from quantum_lake_student.ml import GOOGLE_META_PREDICTION_COLUMNS
from quantum_lake_student.models import RESULTS_PART1, StageResult, code_revision, merge_json_file
from quantum_lake_student.stages import build_ml_tables, load_postgres, prepare_data

GOOGLE_SHOT_SILVER_PATH = "silver/google_qec/shot.parquet"
PREDICTIONS_PATH = Path("results/part2/predictions.parquet")


class TraceMismatch(RuntimeError):
    """A value differs between two layers of a traced record."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _jsonable(value):
    """JSON-friendly copy of a value: bytes as hex, Decimal as str."""
    if isinstance(value, (bytes, memoryview)):
        return bytes(value).hex()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _require(checks: dict[str, bool], name: str, ok: bool) -> None:
    checks[name] = bool(ok)
    if not ok:
        raise TraceMismatch(f"trace check failed: {name}")


def _one_row(conn, sql: str, params: tuple) -> dict:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d.name for d in cur.description]
        rows = cur.fetchall()
    if len(rows) != 1:
        raise TraceMismatch(f"expected exactly one row, got {len(rows)}: {sql} {params}")
    return dict(zip(columns, rows[0]))


def _all_rows(conn, sql: str, params: tuple) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _rows_where(table: pa.Table, column: str, value) -> list[dict]:
    return table.filter(pc.equal(table[column], value)).to_pylist()


# ---------------------------------------------------------------------------
# Bronze: extract the raw record a source_trace row points at
# ---------------------------------------------------------------------------


def csv_record(member_text: str, locator: str) -> str:
    """Data row `locator` (1-based, header excluded) of a syndrome CSV member,
    counted the same way build_syndromes numbers rows."""
    return member_text.splitlines()[int(locator)]


def b8_record(member_bytes: bytes, locator: str, shots: int) -> bytes:
    """Packed row of one shot in a Stim b8 member; locator is "shot_index=N".
    Every shot has the same byte length, so it is the file length / shots."""
    shot_index = int(locator.removeprefix("shot_index="))
    record_bytes, remainder = divmod(len(member_bytes), shots)
    if remainder:
        raise TraceMismatch(f"b8 member length {len(member_bytes)} is not a multiple of {shots} shots")
    return member_bytes[shot_index * record_bytes:(shot_index + 1) * record_bytes]


def zero_one_record(member_text: str, locator: str) -> str:
    """Line of a Stim 01 member; locator is "line=N" (1-based)."""
    return member_text.splitlines()[int(locator.removeprefix("line=")) - 1].strip()


def parse_syndrome_csv_line(line: str) -> tuple[bool, bytes, int]:
    """(label, 16 syndrome bytes round-major, quantity) of one raw CSV row, parsed
    independently of prepare_data so the trace is a real cross-check."""
    label, rest = line.split(",", 1)
    syndromes, quantity = rest.rsplit(",", 1)
    digits = [int(ch) for ch in syndromes if ch in "01"]
    return label.strip() == "1", bytes(digits), int(quantity)


class _BronzeReader:
    """Reads each Bronze object once, verifying its SHA-256 against the trace row."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._archives: dict[str, zipfile.ZipFile] = {}

    def member(self, trace_row: dict) -> bytes:
        name = trace_row["bronze_object"]
        if name not in self._archives:
            data = prepare_data._read_bronze_object(self.settings, name)
            if hashlib.sha256(data).hexdigest() != trace_row["input_sha256"]:
                raise TraceMismatch(f"{name}: Bronze hash differs from source_trace input_sha256")
            self._archives[name] = zipfile.ZipFile(io.BytesIO(data))
        return self._archives[name].read(trace_row["archive_member"])


# ---------------------------------------------------------------------------
# The two traces
# ---------------------------------------------------------------------------


def trace_syndrome(conn, settings: Settings, ml: pa.Table, source_trace: pa.Table,
                   bronze: _BronzeReader) -> dict:
    candidates = ml.filter(pc.and_(pc.equal(ml["data_split"], "test"), ml["logical_error_label"]))
    candidates = candidates.sort_by([("sample_weight", "descending"), ("example_id", "ascending")])
    ml_row = candidates.slice(0, 1).to_pylist()[0]
    checks: dict[str, bool] = {}

    gold_observation = _one_row(
        conn, "SELECT * FROM gold_syndrome_observation WHERE observation_id = %s", (ml_row["example_id"],)
    )
    gold_experiment = _one_row(
        conn, "SELECT * FROM gold_syndrome_experiment WHERE experiment_id = %s", (gold_observation["experiment_id"],)
    )
    gold_pattern = _one_row(
        conn, "SELECT * FROM gold_syndrome_pattern WHERE syndrome_bits = %s", (gold_observation["syndrome_bits"],)
    )
    _require(checks, "ml_matches_gold", (
        ml_row["syndrome_bits"] == bytes(gold_observation["syndrome_bits"])
        and ml_row["logical_error_label"] == gold_observation["logical_error_label"]
        and ml_row["sample_weight"] == gold_observation["quantity"]
        and ml_row["physical_fault_rate"] == float(gold_experiment["physical_fault_rate"])
    ))

    source_record_id = gold_observation["source_record_id"]
    silver_rows = _rows_where(
        load_postgres._read_silver_parquet(settings, prepare_data.SILVER_SYNDROMES_PATH),
        "source_record_id", source_record_id,
    )
    _require(checks, "exactly_one_silver_row", len(silver_rows) == 1)
    silver_row = silver_rows[0]
    _require(checks, "gold_matches_silver", (
        silver_row["syndrome_bits"] == bytes(gold_observation["syndrome_bits"])
        and silver_row["logical_error_label"] == gold_observation["logical_error_label"]
        and silver_row["quantity"] == gold_observation["quantity"]
    ))

    trace_rows = _rows_where(source_trace, "source_record_id", source_record_id)
    _require(checks, "exactly_one_trace_row", len(trace_rows) == 1)
    raw_line = csv_record(bronze.member(trace_rows[0]).decode("utf-8"), trace_rows[0]["record_locator"])
    label, bits, quantity = parse_syndrome_csv_line(raw_line)
    checks["bronze_hash_matches_trace"] = True  # _BronzeReader raised otherwise
    _require(checks, "silver_matches_bronze", (
        bits == silver_row["syndrome_bits"]
        and label == silver_row["logical_error_label"]
        and quantity == silver_row["quantity"]
    ))

    return {
        "example_id": ml_row["example_id"],
        "chosen_as": "test-split example with a logical error and the largest sample_weight",
        "ml": {"table": build_ml_tables.SYNDROME_ML_PATH, "row": ml_row},
        "gold": {
            "resolved_by": "example_id = gold_syndrome_observation.observation_id (primary key)",
            "gold_syndrome_observation": gold_observation,
            "gold_syndrome_experiment": gold_experiment,
            "gold_syndrome_pattern": gold_pattern,
        },
        "silver": {"table": prepare_data.SILVER_SYNDROMES_PATH, "row": silver_row},
        "bronze": {
            "source_trace_rows": trace_rows,
            "raw_records": [{
                "archive_member": trace_rows[0]["archive_member"],
                "record_locator": trace_rows[0]["record_locator"],
                "raw_value": raw_line,
            }],
        },
        "checks": checks,
    }


def trace_google(conn, settings: Settings, ml: pa.Table, source_trace: pa.Table,
                 bronze: _BronzeReader) -> dict:
    candidates = ml.filter(pc.and_(pc.equal(ml["data_split"], "test"), pc.equal(ml["distance"], 3)))
    candidates = candidates.sort_by([("experiment_id", "ascending"), ("shot_index", "ascending")])
    ml_row = candidates.slice(0, 1).to_pylist()[0]
    checks: dict[str, bool] = {}

    key = _one_row(
        conn,
        f"SELECT experiment_id, shot_index FROM {build_ml_tables.GOOGLE_VIEW} WHERE example_id = %s",
        (ml_row["example_id"],),
    )
    gold_shot = _one_row(
        conn, "SELECT * FROM gold_google_shot WHERE experiment_id = %s AND shot_index = %s",
        (key["experiment_id"], key["shot_index"]),
    )
    gold_experiment = _one_row(
        conn, "SELECT * FROM gold_google_experiment WHERE experiment_id = %s", (key["experiment_id"],)
    )
    gold_predictions = _all_rows(
        conn,
        "SELECT * FROM gold_google_decoder_prediction WHERE experiment_id = %s AND shot_index = %s "
        "ORDER BY decoder_name",
        (key["experiment_id"], key["shot_index"]),
    )
    predicted = {r["decoder_name"]: r["predicted_flip"] for r in gold_predictions}
    _require(checks, "ml_matches_gold", (
        ml_row["detector_bits"] == bytes(gold_shot["detector_bits"])
        and ml_row["detector_event_count"] == gold_shot["detector_event_count"]
        and ml_row["actual_observable_flip"] == gold_shot["actual_observable_flip"]
        and all(ml_row[f"{name}_prediction"] == predicted.get(name) for name in load_postgres.DECODER_NAMES)
    ))

    source_record_id = gold_shot["source_record_id"]
    silver = pq.read_table(
        io.BytesIO(_read_lake_bytes(settings, GOOGLE_SHOT_SILVER_PATH)),
        filters=[("source_record_id", "=", source_record_id)],
    )
    silver_rows = silver.to_pylist()
    _require(checks, "exactly_one_silver_row", len(silver_rows) == 1)
    silver_row = silver_rows[0]
    _require(checks, "gold_matches_silver", (
        silver_row["detector_bits"] == bytes(gold_shot["detector_bits"])
        and silver_row["measurement_bits"] == bytes(gold_shot["measurement_bits"])
        and silver_row["actual_observable_flip"] == gold_shot["actual_observable_flip"]
        and all(silver_row[f"{name}_prediction"] == predicted[name] for name in load_postgres.DECODER_NAMES)
    ))

    # one shot is assembled from 8 aligned Bronze members -> 8 trace rows
    trace_rows = sorted(_rows_where(source_trace, "source_record_id", source_record_id),
                        key=lambda r: r["archive_member"])
    _require(checks, "eight_trace_rows_one_per_companion_file", len(trace_rows) == 8)
    shots = gold_experiment["shots"]
    raw_records, bronze_values = [], {}
    for row in trace_rows:
        data = bronze.member(row)
        member = row["archive_member"].rsplit("/", 1)[-1]
        if member.endswith(".b8"):
            value = b8_record(data, row["record_locator"], shots)
        else:
            value = zero_one_record(data.decode("utf-8"), row["record_locator"])
        bronze_values[member] = value
        raw_records.append({
            "archive_member": row["archive_member"], "record_locator": row["record_locator"], "raw_value": value,
        })
    checks["bronze_hash_matches_trace"] = True  # _BronzeReader raised otherwise
    _require(checks, "silver_matches_bronze", (
        bronze_values["detection_events.b8"] == silver_row["detector_bits"]
        and bronze_values["measurements.b8"] == silver_row["measurement_bits"]
        and bronze_values["sweep.b8"] == silver_row["sweep_bits"]
        and (bronze_values["obs_flips_actual.01"] == "1") == silver_row["actual_observable_flip"]
        and all(
            (bronze_values[f"obs_flips_predicted_by_{name}.01"] == "1") == silver_row[f"{name}_prediction"]
            for name in load_postgres.DECODER_NAMES
        )
    ))

    return {
        "example_id": ml_row["example_id"],
        "chosen_as": "lowest test-split shot of the first distance-3 experiment",
        "ml": {"table": build_ml_tables.GOOGLE_ML_PATH, "row": ml_row},
        "gold": {
            "resolved_by": (
                f"{build_ml_tables.GOOGLE_VIEW}.example_id -> (experiment_id, shot_index), "
                "gold_google_shot's primary key"
            ),
            "gold_google_shot": gold_shot,
            "gold_google_experiment": gold_experiment,
            "gold_google_decoder_prediction": gold_predictions,
        },
        "silver": {"table": GOOGLE_SHOT_SILVER_PATH, "row": silver_row},
        "bronze": {"source_trace_rows": trace_rows, "raw_records": raw_records},
        "checks": checks,
    }


def _prediction_rows(example_id: str) -> dict:
    """Part II predictions for this example_id, once results/part2 exists."""
    if not PREDICTIONS_PATH.exists():
        return {"status": "not yet available: Part II writes results/part2/predictions.parquet keyed by example_id"}
    return {"table": str(PREDICTIONS_PATH), "rows": _rows_where(pq.read_table(PREDICTIONS_PATH), "example_id", example_id)}


def _read_lake_bytes(settings: Settings, object_name: str) -> bytes:
    if settings.lake_backend == "local":
        return (settings.local_lake_root / object_name).read_bytes()
    client = connections.minio_client(settings)
    response = client.get_object(settings.s3_bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    result = StageResult(stage="trace_examples", run_id=run_id)

    syndrome_ml = pq.read_table(io.BytesIO(_read_lake_bytes(settings, build_ml_tables.SYNDROME_ML_PATH)))
    google_ml = pq.read_table(io.BytesIO(_read_lake_bytes(settings, build_ml_tables.GOOGLE_ML_PATH)))
    source_trace = pq.read_table(RESULTS_PART1 / "source_trace.parquet")
    bronze = _BronzeReader(settings)

    conn = connections.postgres_connection(settings)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {load_postgres.GOLD_SCHEMA}")
        syndrome = trace_syndrome(conn, settings, syndrome_ml, source_trace, bronze)
        google = trace_google(conn, settings, google_ml, source_trace, bronze)
    finally:
        conn.rollback()  # read-only
        conn.close()

    syndrome["prediction"] = _prediction_rows(syndrome["example_id"])
    google["prediction"] = _prediction_rows(google["example_id"])

    result.input_count = syndrome_ml.num_rows + google_ml.num_rows
    result.output_count = 2
    result.finish()

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    document = {
        "run_id": run_id,
        "code_revision": code_revision(),
        "generated_at": result.finished_at.isoformat(),
        "path": "prediction -> ML row -> Gold record(s) -> Silver row -> source_trace rows -> raw Bronze record(s)",
        "syndrome_trace": syndrome,
        "google_trace": google,
    }
    (RESULTS_PART1 / "trace_examples.json").write_text(json.dumps(_jsonable(document), indent=2) + "\n")
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "trace_examples",
        payload={
            "run_id": run_id,
            "code_revision": document["code_revision"],
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "traced_example_ids": {
                "ml_syndrome_decoder_example": syndrome["example_id"],
                "ml_google_decoder_example": google["example_id"],
            },
        },
    )
    return result
