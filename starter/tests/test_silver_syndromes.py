"""Tests for the qec_syndromes portion of the Silver stage.

Builds small in-memory zips shaped like the real syndromes_dataset.zip so the
parsing/validation logic in prepare_data.build_syndromes() can be exercised
without the course platform or the real Bronze bytes.
"""

from __future__ import annotations

import io
import zipfile
from hashlib import sha256

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.stages import prepare_data as pd

GOOD_ROW = '0,"((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))",100\n'

# build a tiny fake zip
def _zip_bytes(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _syndrome_csv(*rows: str) -> str:
    return pd.SYNDROMES_EXPECTED_HEADER + "\n" + "".join(rows)


def test_known_good_row_parses_correctly():
    data = _zip_bytes({"d-3_pfr-0.000010_nb-10M.csv": _syndrome_csv(GOOD_ROW)})

    table, issues, trace_rows = pd.build_syndromes(data, input_sha256=sha256(data).hexdigest())

    assert [i for i in issues if i.severity == "error"] == []
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["physical_fault_rate"] == 0.00001
    assert row["logical_error_label"] is False
    assert row["quantity"] == 100
    assert row["round_count"] == 4
    assert row["check_count"] == 4
    # round-major, check-second: round 1 has a single set bit at check index 2
    assert row["syndrome_bits"] == bytes([
        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    ])
    assert len(trace_rows) == 1
    assert trace_rows[0].source_record_id == row["source_record_id"]


def test_malformed_tuple_is_rejected_into_issues():
    bad_row = '0,"((0, 0, 0, 0), (0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",100\n'
    data = _zip_bytes({"d-3_pfr-0.000010_nb-10M.csv": _syndrome_csv(GOOD_ROW, bad_row)})

    table, issues, _ = pd.build_syndromes(data, input_sha256=sha256(data).hexdigest())

    assert table.num_rows == 1
    shape_issues = [i for i in issues if i.rule_id == "syndromes.shape"]
    assert len(shape_issues) == 1
    assert shape_issues[0].action == "row_excluded"


def test_non_positive_quantity_is_rejected():
    zero_row = '0,"((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",0\n'
    data = _zip_bytes({"d-3_pfr-0.000010_nb-10M.csv": _syndrome_csv(GOOD_ROW, zero_row)})

    table, issues, _ = pd.build_syndromes(data, input_sha256=sha256(data).hexdigest())

    assert table.num_rows == 1
    quantity_issues = [i for i in issues if i.rule_id == "syndromes.quantity_positive"]
    assert len(quantity_issues) == 1
    assert quantity_issues[0].observed_value == "0"


def test_same_syndrome_both_labels_are_both_kept():
    same_syndrome = "((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    row_label_0 = f'0,"{same_syndrome}",50\n'
    row_label_1 = f'1,"{same_syndrome}",30\n'
    data = _zip_bytes({
        "d-3_pfr-0.000010_nb-10M.csv": _syndrome_csv(row_label_0, row_label_1)
    })

    table, issues, _ = pd.build_syndromes(data, input_sha256=sha256(data).hexdigest())

    assert [i for i in issues if i.severity == "error"] == []
    assert table.num_rows == 2
    labels = sorted(table.column("logical_error_label").to_pylist())
    assert labels == [False, True]
    bit_values = table.column("syndrome_bits").to_pylist()
    assert bit_values[0] == bit_values[1]


def test_accepted_quantity_reconciles_to_filename_hint_on_real_bronze_zip():
    settings = Settings.from_environment()
    try:
        client = minio_client(settings)
        data = client.get_object(
            settings.s3_bucket, "bronze/source=qec_syndromes/syndromes_dataset.zip"
        ).read()
    except Exception:
        import pytest
        pytest.skip("course platform / MinIO not reachable")

    table, issues, _ = pd.build_syndromes(data, input_sha256=sha256(data).hexdigest())

    reconciliation_issues = [
        i for i in issues if i.rule_id == "syndromes.weighted_total_reconciliation"
    ]
    assert reconciliation_issues == []

    total_quantity = sum(table.column("quantity").to_pylist())
    assert abs(total_quantity - 70_000_000) < 0.02 * 70_000_000
