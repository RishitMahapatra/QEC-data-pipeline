"""Parse, check, and connect the supplied QEC data.

Apply documented checks, normalize nested/wide data, connect valid
relationships, and write prepared Parquet tables with chosen column names and
types. Write invalid records and the reason for exclusion to a machine-readable
data-issues output. The project README defines where generated files live.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.models import StageResult

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PART1 = _PROJECT_ROOT / "results" / "part1"



@dataclass(frozen=True)
class DataIssue:
    # why a record was rejected
    source_record_id: str | None
    rule_id: str
    severity: str
    observed_value: str
    action: str
    reason: str


@dataclass(frozen=True)
class TraceRow:
    # trace the row from the original file to its new Silver position
    source_record_id: str
    source_name: str
    bronze_object: str
    archive_member: str
    record_locator: str
    input_sha256: str


def _merge_json_file(path: Path, key: str, payload: dict) -> None:
    existing: dict = json.loads(path.read_text()) if path.exists() else {}
    existing[key] = payload
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))

# constants: define what the correct entries should look like
SYNDROMES_BRONZE_OBJECT = "bronze/source=qec_syndromes/syndromes_dataset.zip"
SYNDROMES_SOURCE_NAME = "qec_syndromes"
SYNDROMES_EXPECTED_HEADER = "labels,syndromes,quantity"
SYNDROMES_FILENAME_RE = re.compile(r"^d-(\d+)_pfr-([\d.]+)_nb-(\w+)\.csv$")

SYNDROMES_SILVER_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("experiment_id", pa.string()),
    ("physical_fault_rate", pa.float64()),
    ("syndrome_bits", pa.binary(16)),  # 4 rounds x 4 checks, one byte per bit
    ("round_count", pa.int32()),
    ("check_count", pa.int32()),
    ("logical_error_label", pa.bool_()),
    ("quantity", pa.int64()),
])


def _syndromes_experiment_id(member: str) -> str:
    # labels rows with which file they came from
    return f"qec_syndromes:{member}"


def _syndromes_record_id(member: str, row_index: int) -> str:
    # Hash of file+position, not random: reruns produce the same id for the
    # same row, so a second run doesn't create duplicates
    return hashlib.sha256(f"{member}:{row_index}".encode()).hexdigest()[:32]


def _syndromes_parse_bits(raw: str) -> list[list[int]] | None:
    # turns the syndromes column text into actual Python numbers and checks
    # if it's really 4 groups of 4 0's/1's; if anything's off, it returns None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, tuple) or len(parsed) != 4:
        return None
    rounds: list[list[int]] = []
    for round_tuple in parsed:
        if not isinstance(round_tuple, tuple) or len(round_tuple) != 4:
            return None
        if not all(bit in (0, 1) for bit in round_tuple):
            return None
        rounds.append(list(round_tuple))
    return rounds


def _syndromes_pack_bits(rounds: list[list[int]]) -> bytes:
    # takes syndromes column numbers and flattens them into 16 single bytes in
    # order, which is what the output table expects
    return bytes(bit for round_bits in rounds for bit in round_bits)


def build_syndromes(bronze_bytes: bytes, input_sha256: str) -> tuple[
    pa.Table, list[DataIssue], list[TraceRow]
]:
    # opens the zip and finds every .csv file
    records: list[dict] = []
    issues: list[DataIssue] = []
    trace_rows: list[TraceRow] = []

    with zipfile.ZipFile(io.BytesIO(bronze_bytes)) as archive:
        csv_members = sorted(n for n in archive.namelist() if n.endswith(".csv"))
        # checks if the filename and header look right
        for member in csv_members:
            match = SYNDROMES_FILENAME_RE.match(member)
            if not match:
                issues.append(DataIssue(
                    source_record_id=None,
                    rule_id="syndromes.filename_format",
                    severity="error",
                    observed_value=member,
                    action="file_skipped",
                    reason="filename does not match d-<distance>_pfr-<rate>_nb-<n>.csv",
                ))
                continue
            fault_rate = float(match.group(2))
            experiment_id = _syndromes_experiment_id(member)

            with archive.open(member) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8")
                header = next(text).strip()
                if header != SYNDROMES_EXPECTED_HEADER:
                    issues.append(DataIssue(
                        source_record_id=None,
                        rule_id="syndromes.header",
                        severity="error",
                        observed_value=header,
                        action="file_skipped",
                        reason=f"expected header '{SYNDROMES_EXPECTED_HEADER}'",
                    ))
                    continue

                # reads it and for each line it splits out the 3 fields
                accepted_quantity_total = 0
                for row_index, line in enumerate(text, start=1):
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    # quantity is the last comma field; syndromes is quoted and
                    # may itself contain commas, so split from the right
                    label_str, rest = line.split(",", 1)
                    syndromes_str, quantity_str = rest.rsplit(",", 1)
                    syndromes_str = syndromes_str.strip('"')

                    # records a trace entry for the row
                    record_id = _syndromes_record_id(member, row_index)
                    trace_rows.append(TraceRow(
                        source_record_id=record_id,
                        source_name=SYNDROMES_SOURCE_NAME,
                        bronze_object=SYNDROMES_BRONZE_OBJECT,
                        archive_member=member,
                        record_locator=str(row_index),
                        input_sha256=input_sha256,
                    ))

                    # runs the validity checks
                    rounds = _syndromes_parse_bits(syndromes_str)
                    try:
                        quantity = int(quantity_str)
                    except ValueError:
                        quantity = None
                    label = label_str.strip()

                    if rounds is None:
                        issues.append(DataIssue(
                            source_record_id=record_id,
                            rule_id="syndromes.shape",
                            severity="error",
                            observed_value=syndromes_str,
                            action="row_excluded",
                            reason="syndromes value is not a valid 4x4 binary tuple",
                        ))
                        continue
                    if quantity is None or quantity <= 0:
                        issues.append(DataIssue(
                            source_record_id=record_id,
                            rule_id="syndromes.quantity_positive",
                            severity="error",
                            observed_value=quantity_str,
                            action="row_excluded",
                            reason="quantity must be a positive integer",
                        ))
                        continue
                    if label not in ("0", "1"):
                        issues.append(DataIssue(
                            source_record_id=record_id,
                            rule_id="syndromes.label_domain",
                            severity="error",
                            observed_value=label,
                            action="row_excluded",
                            reason="labels must be 0 or 1",
                        ))
                        continue

                    
                    accepted_quantity_total += quantity
                    records.append({
                        "source_record_id": record_id,
                        "experiment_id": experiment_id,
                        "physical_fault_rate": fault_rate,
                        "syndrome_bits": _syndromes_pack_bits(rounds),
                        "round_count": 4,
                        "check_count": 4,
                        "logical_error_label": bool(int(label)),
                        "quantity": quantity,
                    })

                # checks that the total quantity of everything kept
                # roughly matches what the filename promised as to 
                # not silently lose or duplicate anything
                nb_hint = match.group(3)  
                if nb_hint.endswith("M"):
                    expected_total = float(nb_hint[:-1]) * 1_000_000
                    if abs(accepted_quantity_total - expected_total) > 0.02 * expected_total:
                        issues.append(DataIssue(
                            source_record_id=None,
                            rule_id="syndromes.weighted_total_reconciliation",
                            severity="warning",
                            observed_value=str(accepted_quantity_total),
                            action="file_flagged",
                            reason=(
                                f"{member}: accepted weighted total "
                                f"{accepted_quantity_total} vs filename hint "
                                f"{expected_total}"
                            ),
                        ))

    # builds one clean table from everything that passed
    table = pa.Table.from_pylist(records, schema=SYNDROMES_SILVER_SCHEMA)
    return table, issues, trace_rows


def _sha256(data: bytes) -> str:
    # produces an identification for the content
    return hashlib.sha256(data).hexdigest()


def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    client = minio_client(settings)
    result = StageResult(stage="prepare_data", run_id=run_id)

    all_issues: list[DataIssue] = []
    all_trace_rows: list[TraceRow] = []

    syndromes_bytes = client.get_object(settings.s3_bucket, SYNDROMES_BRONZE_OBJECT).read()
    syndromes_table, syndromes_issues, syndromes_trace = build_syndromes(
        syndromes_bytes, input_sha256=_sha256(syndromes_bytes)
    )
    buffer = io.BytesIO()
    pq.write_table(syndromes_table, buffer)
    buffer.seek(0)
    client.put_object(
        settings.s3_bucket,
        "silver/qec_syndromes/syndrome_observation.parquet",
        buffer,
        length=buffer.getbuffer().nbytes,
    )
    all_issues.extend(syndromes_issues)
    all_trace_rows.extend(syndromes_trace)
    result.output_count += syndromes_table.num_rows

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)

    # overwritten wholesale each run
    issues_table = pa.Table.from_pylist([
        {
            "issue_id": f"{issue.rule_id}:{issue.source_record_id or 'file'}:{n}",
            "run_id": run_id,
            "source_record_id": issue.source_record_id,
            "rule_id": issue.rule_id,
            "severity": issue.severity,
            "observed_value": issue.observed_value,
            "action": issue.action,
            "reason": issue.reason,
        }
        for n, issue in enumerate(all_issues)
    ])
    pq.write_table(issues_table, RESULTS_PART1 / "data_issues.parquet")

    trace_table = pa.Table.from_pylist([row.__dict__ for row in all_trace_rows])
    pq.write_table(trace_table, RESULTS_PART1 / "source_trace.parquet")

    result.issue_count = len(all_issues)
    result.finish()

    _merge_json_file(
        RESULTS_PART1 / "run.json",
        key="prepare_data",
        payload={
            "run_id": run_id,
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "output_count": result.output_count,
            "issue_count": result.issue_count,
        },
    )
    _merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        key="prepare_data",
        payload={
            "accepted": result.output_count,
            "rejected": result.issue_count,
        },
    )

    return result
