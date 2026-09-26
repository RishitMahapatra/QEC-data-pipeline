"""Tests for the Silver prepare_data stage: google_qec."""

from __future__ import annotations

import io
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from quantum_lake_student.config import Settings
from quantum_lake_student.formats import iter_b8_records
from quantum_lake_student.stages import prepare_data as prep


EXP = "surface_code_bX_d3_r25_center_3_5"
PROPS = {
    "type": "surface_code_memory_experiment",
    "basis": "X",
    "rounds": 25,
    "distance": 3,
    "shots": 4,
    "center_data_qubit_row": 3,
    "center_data_qubit_col": 5,
    "circuit_measurements": 10,  # 2 bytes per shot, 6 padding bits
    "circuit_sweep_bits": 9,  # 2 bytes per shot, 7 padding bits
    "circuit_detectors": 8,  # 1 byte per shot, no padding
}
MEASUREMENTS = [bytes([0x10 + i, i % 4]) for i in range(4)]
SWEEP = [bytes([0xA0 + i, i % 2]) for i in range(4)]
DETECTORS = [bytes([b]) for b in (0b00000000, 0b00000001, 0b00000011, 0b10000001)]
ACTUAL = [0, 1, 1, 0]
PREDICTED = {
    "belief_matching": [0, 1, 0, 0],
    "correlated_matching": [1, 1, 1, 0],
    "pymatching": [0, 0, 1, 1],
    "tensor_network_contraction": [0, 1, 1, 0],
}


def _flags(values: list[int]) -> bytes:
    return "".join(f"{v}\n" for v in values).encode()


def default_files() -> dict[str, bytes]:
    files = {
        "measurements.b8": b"".join(MEASUREMENTS),
        "sweep.b8": b"".join(SWEEP),
        "detection_events.b8": b"".join(DETECTORS),
        "obs_flips_actual.01": _flags(ACTUAL),
    }
    for name, values in PREDICTED.items():
        files[f"obs_flips_predicted_by_{name}.01"] = _flags(values)
    return files


def make_archive_bytes(*, props=None, remove_props=(), replace=None, drop=(), exp_dir=EXP) -> bytes:
    merged = {**PROPS, **(props or {})}
    for key in remove_props:
        merged.pop(key)
    files = {**default_files(), **(replace or {})}
    for name in drop:
        files.pop(name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        out.writestr(f"{exp_dir}/properties.yml", yaml.safe_dump(merged))
        for name, data in files.items():
            out.writestr(f"{exp_dir}/{name}", data)
    return buffer.getvalue()


def make_archive(**kwargs) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(make_archive_bytes(**kwargs)))


def build(archive, run_id="test-run"):
    return prep.build_google_table(
        archive, bronze_object="bronze/source=google_qec/x.zip", input_sha256="abc123", run_id=run_id
    )


def test_builds_the_documented_tables() -> None:
    silver = build(make_archive())

    assert silver.experiment.schema == prep.GOOGLE_EXPERIMENT_SCHEMA
    assert silver.shot.schema == prep.GOOGLE_SHOT_SCHEMA
    assert silver.trace.schema == prep.TRACE_SCHEMA
    assert silver.issues == []
    assert silver.shots_read == 4

    experiment = silver.experiment.to_pylist()
    assert len(experiment) == 1
    assert {k: v for k, v in experiment[0].items() if k != "source_record_id"} == {
        "experiment_id": EXP,
        "basis": "X",
        "distance": 3,
        "rounds": 25,
        "shots": 4,
        "center_row": 3,
        "center_col": 5,
        "measurement_count": 10,
        "detector_count": 8,
    }

    shots = silver.shot.to_pylist()
    assert [s["shot_index"] for s in shots] == [0, 1, 2, 3]
    assert [s["measurement_bits"] for s in shots] == MEASUREMENTS  # packed rows, unchanged
    assert [s["sweep_bits"] for s in shots] == SWEEP
    assert [s["detector_bits"] for s in shots] == DETECTORS
    assert [s["detector_event_count"] for s in shots] == [0, 1, 2, 2]
    assert [s["actual_observable_flip"] for s in shots] == [bool(v) for v in ACTUAL]
    for name, values in PREDICTED.items():
        assert [s[f"{name}_prediction"] for s in shots] == [bool(v) for v in values]
    assert all(s["experiment_id"] == EXP for s in shots)


def test_every_row_is_traced_and_a_shot_has_eight_trace_rows() -> None:
    silver = build(make_archive())
    trace = silver.trace.to_pylist()
    assert len(trace) == 1 + 4 * 8

    traced_ids = {t["source_record_id"] for t in trace}
    silver_ids = {r["source_record_id"] for r in silver.experiment.to_pylist()} | {
        r["source_record_id"] for r in silver.shot.to_pylist()
    }
    assert silver_ids == traced_ids

    shot2 = silver.shot.to_pylist()[2]["source_record_id"]
    rows = [t for t in trace if t["source_record_id"] == shot2]
    assert len(rows) == 8
    assert {r["archive_member"].rsplit("/", 1)[1]: r["record_locator"] for r in rows} == {
        "measurements.b8": "shot_index=2",
        "sweep.b8": "shot_index=2",
        "detection_events.b8": "shot_index=2",
        "obs_flips_actual.01": "line=3",
        "obs_flips_predicted_by_belief_matching.01": "line=3",
        "obs_flips_predicted_by_correlated_matching.01": "line=3",
        "obs_flips_predicted_by_pymatching.01": "line=3",
        "obs_flips_predicted_by_tensor_network_contraction.01": "line=3",
    }
    assert {t["source_name"] for t in trace} == {"google_qec"}
    assert {t["input_sha256"] for t in trace} == {"abc123"}


def test_second_run_gives_identical_records_and_ids() -> None:
    first = build(make_archive(), run_id="run-1")
    second = build(make_archive(), run_id="run-2")
    assert first.experiment.equals(second.experiment)
    assert first.shot.equals(second.shot)
    assert first.trace.equals(second.trace)
    ids = first.shot["source_record_id"].to_pylist()
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize(
    ("label", "kwargs", "message"),
    [
        ("name vs properties", {"props": {"distance": 5}}, "name says"),
        ("missing property", {"remove_props": ("circuit_detectors",)}, "missing key"),
        ("non-integer property", {"props": {"shots": "many"}}, "non-negative integer"),
        ("zero shots", {"props": {"shots": 0}}, "shots must be positive"),
        ("truncated b8", {"replace": {"measurements.b8": b"".join(MEASUREMENTS)[:-1]}}, "expected 8"),
        ("extra b8 byte", {"replace": {"detection_events.b8": b"".join(DETECTORS) + b"\x00"}}, "expected 4"),
        ("actual 01 short", {"replace": {"obs_flips_actual.01": _flags(ACTUAL[:-1])}}, "3 lines, expected 4"),
        (
            "predicted 01 not binary",
            {"replace": {"obs_flips_predicted_by_pymatching.01": b"0\n2\n1\n0\n"}},
            "other than 0 or 1",
        ),
        (
            "predicted 01 long",
            {"replace": {"obs_flips_predicted_by_belief_matching.01": _flags(PREDICTED["belief_matching"] + [0])}},
            "5 lines, expected 4",
        ),
        ("missing companion file", {"drop": ("obs_flips_predicted_by_pymatching.01",)}, "missing"),
        ("missing sweep file", {"drop": ("sweep.b8",)}, "missing"),
    ],
)
def test_unusable_experiment_stops_the_run(label, kwargs, message) -> None:
    with pytest.raises(prep.GoogleSilverError, match=message):
        build(make_archive(**kwargs))


def test_archive_without_experiments_stops_the_run() -> None:
    empty = zipfile.ZipFile(io.BytesIO(_empty_zip()))
    with pytest.raises(prep.GoogleSilverError, match="no experiment"):
        build(empty)


def _empty_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        out.writestr("readme.txt", "nothing here")
    return buffer.getvalue()


def test_shot_with_dirty_padding_is_excluded_and_reported() -> None:
    measurements = list(MEASUREMENTS)
    measurements[1] = bytes([0x11, 0b00000100])  # bit 10 is padding for a 10-bit row
    silver = build(make_archive(replace={"measurements.b8": b"".join(measurements)}))

    assert [s["shot_index"] for s in silver.shot.to_pylist()] == [0, 2, 3]
    assert len(silver.issues) == 1
    issue = silver.issues[0]
    assert issue["rule_id"] == "google.b8_padding_bits_zero"
    assert issue["action"] == "excluded_from_silver"
    assert issue["run_id"] == "test-run"
    assert issue["source_record_id"] == prep.google_shot_source_record_id(EXP, 1)
    assert issue["observed_value"] == f"{EXP}/measurements.b8 shot_index=1"

    # rows read = rows accepted + rows sent to data_issues
    assert silver.shots_read == silver.shot.num_rows + len(silver.issues)
    # no trace rows for the excluded shot
    assert issue["source_record_id"] not in set(silver.trace["source_record_id"].to_pylist())
    assert len(silver.trace) == 1 + 3 * 8


def test_padding_is_the_high_bits_of_the_last_byte() -> None:
    # 10-bit records: bits 0-9 are data, bits 10-15 are padding (little-endian in each byte)
    two_shots = bytes([0x01, 0x02]) + bytes([0x01, 0x04])  # bit 9 set / bit 10 set
    assert prep.dirty_padding_shots(two_shots, bits_per_shot=10) == [1]
    # byte-aligned rows have no padding to check
    assert prep.dirty_padding_shots(b"\xff\xff", bits_per_shot=8) == []
    assert prep.dirty_padding_shots(b"", bits_per_shot=0) == []


def test_packed_bits_are_little_endian_within_each_byte() -> None:
    silver = build(make_archive())
    row = silver.shot.to_pylist()[1]  # measurements 0x11 0x01, detectors 0b00000001
    measurement_bits = next(iter_b8_records(row["measurement_bits"], bits_per_record=10))
    assert [i for i, bit in enumerate(measurement_bits) if bit] == [0, 4, 8]  # 0x11 -> bits 0,4; 0x01 -> bit 8
    detector_bits = next(iter_b8_records(row["detector_bits"], bits_per_record=8))
    assert [i for i, bit in enumerate(detector_bits) if bit] == [0]
    assert row["detector_event_count"] == sum(detector_bits)


def test_experiment_without_sweep_bits_gets_empty_sweep_values() -> None:
    silver = build(make_archive(props={"circuit_sweep_bits": 0}, replace={"sweep.b8": b""}))
    assert [s["sweep_bits"] for s in silver.shot.to_pylist()] == [b""] * 4


def test_several_experiments_are_kept_separate() -> None:
    other = "surface_code_bX_d3_r25_center_5_3"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        for exp_dir, props in ((EXP, PROPS), (other, {**PROPS, "center_data_qubit_row": 5, "center_data_qubit_col": 3})):
            out.writestr(f"{exp_dir}/properties.yml", yaml.safe_dump(props))
            for name, data in default_files().items():
                out.writestr(f"{exp_dir}/{name}", data)
    silver = build(zipfile.ZipFile(io.BytesIO(buffer.getvalue())))

    assert [r["experiment_id"] for r in silver.experiment.to_pylist()] == sorted([EXP, other])
    assert silver.shot.num_rows == 8
    assert silver.shots_read == 8
    assert len(set(silver.shot["source_record_id"].to_pylist())) == 8  # same shot_index, different experiment


def test_load_google_archive_reads_bronze_and_hashes_the_bytes(monkeypatch) -> None:
    data = make_archive_bytes()

    class FakeResponse:
        def read(self):
            return data

        def close(self):
            pass

        def release_conn(self):
            pass

    class FakeClient:
        def get_object(self, bucket, name):
            assert name == "bronze/source=google_qec/g.zip"
            return FakeResponse()

    monkeypatch.setattr(
        prep,
        "bronze_inventory",
        lambda settings: [("bronze/source=qasmbench/q.zip", 1), ("bronze/source=google_qec/g.zip", len(data))],
    )
    monkeypatch.setattr(prep, "minio_client", lambda settings: FakeClient())

    import hashlib

    zf, object_name, sha = prep.load_google_archive(Settings.from_environment())
    assert object_name == "bronze/source=google_qec/g.zip"
    assert sha == hashlib.sha256(data).hexdigest()
    assert prep.google_experiment_dirs(zf.namelist()) == [EXP]


def test_load_google_archive_stops_when_bronze_has_no_google_object(monkeypatch) -> None:
    monkeypatch.setattr(prep, "bronze_inventory", lambda settings: [("bronze/source=qasmbench/q.zip", 1)])
    with pytest.raises(prep.GoogleSilverError, match="no google_qec object"):
        prep.load_google_archive(Settings.from_environment())


# ---------------------------------------------------------------------------
# run(): writes the Silver tables and the shared results files. The local lake
# backend is used, so no MinIO is needed.
# ---------------------------------------------------------------------------

BRONZE_NAME = "bronze/source=google_qec/google-surface-code-curated.zip"

# A single valid, issue-free syndromes file: run() also builds qec_syndromes
# now, so every test that calls run() needs Bronze bytes for it too. The
# "nb-2" filename hint doesn't end in "M", so the weighted-total
# reconciliation check is skipped and this one row never becomes an issue.
SYNDROMES_ROW = '0,"((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",5\n'


def _syndromes_archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        out.writestr(
            "d-3_pfr-0.000010_nb-2.csv",
            "labels,syndromes,quantity\n" + SYNDROMES_ROW,
        )
    return buffer.getvalue()


# A single valid, minimal circuit: run() also builds qasmbench now, so every
# test that calls run() needs Bronze bytes for it too. No cx/if_else means no
# stabilizer_check or conditional_correction rows, just one circuit row.
QASM_BENCHMARK = "sample_circuit"
QASM_SOURCE = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n"


def _qasmbench_archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        out.writestr(f"{QASM_BENCHMARK}/{QASM_BENCHMARK}.qasm", QASM_SOURCE)
    return buffer.getvalue()


def stage_run(tmp_path, monkeypatch, archive_bytes: bytes):
    lake = tmp_path / "lake"
    (lake / BRONZE_NAME).parent.mkdir(parents=True)
    (lake / BRONZE_NAME).write_bytes(archive_bytes)
    (lake / prep.SYNDROMES_BRONZE_OBJECT).parent.mkdir(parents=True)
    (lake / prep.SYNDROMES_BRONZE_OBJECT).write_bytes(_syndromes_archive_bytes())
    (lake / prep.BRONZE_OBJECT).parent.mkdir(parents=True)
    (lake / prep.BRONZE_OBJECT).write_bytes(_qasmbench_archive_bytes())
    results = tmp_path / "results" / "part1"
    monkeypatch.setenv("LAKE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LAKE_ROOT", str(lake))
    monkeypatch.setattr(prep, "RESULTS_PART1", results)
    return lake, results


def test_run_writes_silver_tables_and_results(tmp_path, monkeypatch) -> None:
    lake, results = stage_run(tmp_path, monkeypatch, make_archive_bytes())

    result = prep.run("run-1")

    experiment = pq.read_table(lake / prep.SILVER_EXPERIMENT_PATH)
    shot = pq.read_table(lake / prep.SILVER_SHOT_PATH)
    assert experiment.schema == prep.GOOGLE_EXPERIMENT_SCHEMA
    assert shot.schema == prep.GOOGLE_SHOT_SCHEMA
    assert (experiment.num_rows, shot.num_rows) == (1, 4)

    trace = pq.read_table(results / "source_trace.parquet")
    # google_qec (1 experiment + 4 shots x 8 trace rows) + qec_syndromes (1) + qasmbench (1)
    assert trace.schema == prep.TRACE_SCHEMA and trace.num_rows == 1 + 4 * 8 + 1 + 1
    issues = pq.read_table(results / "data_issues.parquet")
    assert issues.schema == prep.DATA_ISSUES_SCHEMA and issues.num_rows == 0

    counts = json.loads((results / "row_counts.json").read_text())["prepare_data"]["google_qec"]
    assert counts["shot"] == {"read": 4, "accepted": 4, "rejected": 0}
    assert counts["experiment"] == {"read": 1, "accepted": 1, "rejected": 0}
    run_info = json.loads((results / "run.json").read_text())["prepare_data"]["google_qec"]
    assert run_info["run_id"] == "run-1"
    assert list(run_info["input_hashes"]) == ["google_qec"]
    assert run_info["output_row_counts"][prep.SILVER_SHOT_PATH] == 4

    syndromes = pq.read_table(lake / prep.SILVER_SYNDROMES_PATH)
    assert syndromes.schema == prep.SYNDROMES_SILVER_SCHEMA and syndromes.num_rows == 1
    syndromes_counts = json.loads((results / "row_counts.json").read_text())["prepare_data"]["qec_syndromes"]
    assert syndromes_counts == {"read": 1, "accepted": 1, "rejected": 0}

    circuit = pq.read_table(lake / "silver/qasmbench/circuit.parquet")
    assert circuit.num_rows == 1
    qasmbench_counts = json.loads((results / "row_counts.json").read_text())["prepare_data"]["qasmbench"]
    assert qasmbench_counts["circuit"] == {"read": 1, "accepted": 1, "rejected": 0}

    # google_qec (1 experiment + 4 shots) + qec_syndromes (1 row) + qasmbench (1 circuit)
    assert (result.stage, result.input_count, result.output_count, result.issue_count) == ("prepare_data", 7, 7, 0)


def test_run_twice_leaves_same_records_and_no_duplicates(tmp_path, monkeypatch) -> None:
    lake, results = stage_run(tmp_path, monkeypatch, make_archive_bytes())

    prep.run("run-1")
    first = {p: pq.read_table(p) for p in (lake / prep.SILVER_SHOT_PATH, lake / prep.SILVER_EXPERIMENT_PATH, results / "source_trace.parquet")}
    prep.run("run-2")

    for path, table in first.items():
        assert pq.read_table(path).equals(table), path.name


def test_run_reports_rejected_shots_in_data_issues(tmp_path, monkeypatch) -> None:
    measurements = list(MEASUREMENTS)
    measurements[3] = bytes([0x13, 0b00001000])
    lake, results = stage_run(
        tmp_path, monkeypatch, make_archive_bytes(replace={"measurements.b8": b"".join(measurements)})
    )

    result = prep.run("run-1")

    issues = pq.read_table(results / "data_issues.parquet").to_pylist()
    assert [i["rule_id"] for i in issues] == ["google.b8_padding_bits_zero"]
    assert pq.read_table(lake / prep.SILVER_SHOT_PATH).num_rows == 3
    counts = json.loads((results / "row_counts.json").read_text())["prepare_data"]["google_qec"]
    assert counts["shot"] == {"read": 4, "accepted": 3, "rejected": 1}
    assert result.issue_count == 1

    prep.run("run-2")  # rerun replaces Google's issue rows instead of adding to them
    assert pq.read_table(results / "data_issues.parquet").num_rows == 1


def test_run_keeps_other_sources_rows_in_shared_results_files(tmp_path, monkeypatch) -> None:
    # A fictitious source stands in for one run() doesn't build (all three
    # real sources are covered by other tests): proves run() only ever
    # replaces its own sources' rows in the shared results files.
    lake, results = stage_run(tmp_path, monkeypatch, make_archive_bytes())
    results.mkdir(parents=True)
    other_trace = {
        "source_record_id": "other-1", "source_name": "other_source", "bronze_object": "b",
        "archive_member": "m", "record_locator": "l", "input_sha256": "s",
    }
    other_issue = {
        "issue_id": "i-1", "run_id": "r", "source_record_id": None, "rule_id": "other_source.bad_shape",
        "severity": "error", "observed_value": "x", "action": "excluded_from_silver", "reason": "why",
    }
    pq.write_table(pa.Table.from_pylist([other_trace], schema=prep.TRACE_SCHEMA), results / "source_trace.parquet")
    pq.write_table(pa.Table.from_pylist([other_issue], schema=prep.DATA_ISSUES_SCHEMA), results / "data_issues.parquet")
    (results / "row_counts.json").write_text(json.dumps({"prepare_data": {"other_source": {"kept": True}}}))

    prep.run("run-1")
    prep.run("run-2")

    trace = pq.read_table(results / "source_trace.parquet").to_pylist()
    assert [t["source_record_id"] for t in trace if t["source_name"] == "other_source"] == ["other-1"]
    assert sum(t["source_name"] == "google_qec" for t in trace) == 1 + 4 * 8
    issue_ids = [i["issue_id"] for i in pq.read_table(results / "data_issues.parquet").to_pylist()]
    assert "i-1" in issue_ids
    row_counts = json.loads((results / "row_counts.json").read_text())["prepare_data"]
    assert row_counts["other_source"] == {"kept": True}
    assert "google_qec" in row_counts and "qec_syndromes" in row_counts and "qasmbench" in row_counts


def test_run_writes_nothing_when_a_check_fails(tmp_path, monkeypatch) -> None:
    truncated = make_archive_bytes(replace={"measurements.b8": b"".join(MEASUREMENTS)[:-1]})
    lake, results = stage_run(tmp_path, monkeypatch, truncated)

    with pytest.raises(prep.GoogleSilverError):
        prep.run("run-1")

    assert not (lake / "silver").exists()
    assert not results.exists()
