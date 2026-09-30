"""Tests for trace_examples.py's Bronze hop.

The riskiest part of a trace is the last step: a source_trace row's
record_locator must point at exactly the raw record prepare_data built the
Silver row from. These tests build Silver with the real prepare_data
functions from tiny in-memory archives, then extract every traced record
again with the trace helpers and check it gives back the Silver values.
"""

from __future__ import annotations

import hashlib
import io
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from quantum_lake_student.stages import prepare_data as prep
from quantum_lake_student.stages import trace_examples as te

SYNDROME_ROWS = [
    '0,"((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))",100\n',
    '1,"((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))",30\n',
    '1,"((1, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))",7\n',
]


def _zip(members: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_csv_locator_points_at_the_row_silver_was_built_from() -> None:
    member = "d-3_pfr-0.005000_nb-10M.csv"
    text = prep.SYNDROMES_EXPECTED_HEADER + "\n" + "".join(SYNDROME_ROWS)
    data = _zip({member: text})
    table, _, trace_rows = prep.build_syndromes(data, input_sha256=hashlib.sha256(data).hexdigest())

    silver = {r["source_record_id"]: r for r in table.to_pylist()}
    for trace in trace_rows:
        line = te.csv_record(text, trace.record_locator)
        label, bits, quantity = te.parse_syndrome_csv_line(line)
        row = silver[trace.source_record_id]
        assert (label, bits, quantity) == (row["logical_error_label"], row["syndrome_bits"], row["quantity"])


EXP = "surface_code_bX_d3_r25_center_3_5"
SHOTS = 3


def _google_archive() -> bytes:
    props = {
        "type": "surface_code_memory_experiment", "basis": "X", "rounds": 25, "distance": 3,
        "shots": SHOTS, "center_data_qubit_row": 3, "center_data_qubit_col": 5,
        "circuit_measurements": 10, "circuit_sweep_bits": 9, "circuit_detectors": 8,
    }
    files = {
        "properties.yml": yaml.safe_dump(props),
        "measurements.b8": bytes([0x11, 0x01, 0x22, 0x02, 0x33, 0x03]),
        "sweep.b8": bytes([0xA0, 0x00, 0xA1, 0x01, 0xA2, 0x00]),
        "detection_events.b8": bytes([0b00000000, 0b00000101, 0b10000001]),
        "obs_flips_actual.01": "0\n1\n1\n",
        "obs_flips_predicted_by_belief_matching.01": "0\n1\n0\n",
        "obs_flips_predicted_by_correlated_matching.01": "1\n1\n1\n",
        "obs_flips_predicted_by_pymatching.01": "0\n0\n1\n",
        "obs_flips_predicted_by_tensor_network_contraction.01": "0\n1\n1\n",
    }
    return _zip({f"{EXP}/{name}": content for name, content in files.items()})


def test_b8_and_01_locators_point_at_the_shot_silver_was_built_from() -> None:
    data = _google_archive()
    archive = zipfile.ZipFile(io.BytesIO(data))
    silver = prep.build_google_table(archive, bronze_object="bronze/x.zip", input_sha256="h", run_id="r")

    shots = {r["source_record_id"]: r for r in silver.shot.to_pylist()}
    checked = 0
    for trace in silver.trace.to_pylist():
        if trace["source_record_id"] not in shots:
            continue  # the experiment's properties.yml trace row
        row = shots[trace["source_record_id"]]
        member = trace["archive_member"].rsplit("/", 1)[-1]
        raw = archive.read(trace["archive_member"])
        if member.endswith(".b8"):
            column = {"detection_events.b8": "detector_bits", "measurements.b8": "measurement_bits",
                      "sweep.b8": "sweep_bits"}[member]
            assert te.b8_record(raw, trace["record_locator"], SHOTS) == row[column]
        else:
            column = ("actual_observable_flip" if member == "obs_flips_actual.01"
                      else member.removeprefix("obs_flips_predicted_by_").removesuffix(".01") + "_prediction")
            assert (te.zero_one_record(raw.decode(), trace["record_locator"]) == "1") == row[column]
        checked += 1
    assert checked == SHOTS * 8  # 8 companion files per shot


def test_b8_record_rejects_a_member_that_is_not_whole_shots() -> None:
    with pytest.raises(te.TraceMismatch, match="not a multiple"):
        te.b8_record(bytes(7), "shot_index=0", shots=3)


def test_bronze_reader_rejects_an_object_whose_hash_differs(monkeypatch) -> None:
    data = _zip({"a.csv": "x"})
    monkeypatch.setattr(prep, "_read_bronze_object", lambda _settings, _name: data)
    reader = te._BronzeReader(settings=None)
    row = {"bronze_object": "bronze/a.zip", "archive_member": "a.csv", "input_sha256": "0" * 64}
    with pytest.raises(te.TraceMismatch, match="hash"):
        reader.member(row)
    good = {**row, "input_sha256": hashlib.sha256(data).hexdigest()}
    assert reader.member(good) == b"x"


def test_require_records_the_check_and_raises_on_failure() -> None:
    checks: dict[str, bool] = {}
    te._require(checks, "ok", True)
    with pytest.raises(te.TraceMismatch, match="gold_matches_silver"):
        te._require(checks, "gold_matches_silver", False)
    assert checks == {"ok": True, "gold_matches_silver": False}


def test_jsonable_turns_bytes_and_decimals_into_json_values() -> None:
    from decimal import Decimal
    assert te._jsonable({"b": b"\x01\xff", "d": Decimal("0.005"), "l": [b"\x00"]}) == {
        "b": "01ff", "d": "0.005", "l": ["00"],
    }


def test_prediction_step_reports_missing_part2_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(te, "PREDICTIONS_PATH", tmp_path / "predictions.parquet")
    assert "not yet available" in te._prediction_rows("abc")["status"]

    pq.write_table(
        pa.table({"example_id": ["abc", "def"], "model": ["m", "m"]}), tmp_path / "predictions.parquet"
    )
    assert te._prediction_rows("abc")["rows"] == [{"example_id": "abc", "model": "m"}]
