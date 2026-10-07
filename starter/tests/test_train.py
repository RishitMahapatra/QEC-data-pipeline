"""Tests for the Part II training stage (Tasks A, B and C).

They use small hand-made tables, so no database, MinIO or real data is needed.
"""

from __future__ import annotations

import io
import json
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.ml import unpack_little_endian_bits
from quantum_lake_student.stages import train

ZERO = bytes(16)


def _bits(*fired: int) -> bytes:
    """This function builds a 16-byte syndrome with the given bits set to 1."""
    return bytes(1 if i in fired else 0 for i in range(16))


# (example_id, fault rate, syndrome, label, weight)
ROWS = [
    ("tr-0", 0.0001, ZERO, False, 900),
    ("tr-1", 0.0001, _bits(0), True, 60),
    ("tr-2", 0.0001, _bits(0), False, 40),
    ("tr-3", 0.01, _bits(2), True, 70),
    ("tr-4", 0.01, _bits(2), False, 30),
    ("tr-5", 0.01, ZERO, False, 500),
    ("va-0", 0.0005, ZERO, False, 400),
    ("va-1", 0.0005, _bits(0), True, 20),
    ("va-2", 0.0005, _bits(2), False, 10),
    ("te-0", 0.005, ZERO, False, 300),
    ("te-1", 0.005, _bits(0), True, 50),
    ("te-2", 0.005, _bits(2), True, 25),
]


def make_table(rows=ROWS, **overrides) -> pa.Table:
    """This function builds a syndrome table with the contract schema."""
    records = [
        {
            "example_id": example_id,
            "experiment_id": f"qec_syndromes:rate-{rate}",
            "physical_fault_rate": rate,
            "syndrome_bits": bits,
            "round_count": 4,
            "check_count": 4,
            "logical_error_label": label,
            "sample_weight": weight,
            "data_split": train.syndrome_data_split(rate),
        }
        for example_id, rate, bits, label, weight in rows
    ]
    for record in records:
        record.update(overrides)
    return pa.Table.from_pylist(records, schema=train.SYNDROME_INPUT_SCHEMA)


def _without_timings(metrics: dict) -> dict:
    return {
        **metrics,
        "models": {
            model_id: {k: v for k, v in m.items() if not k.endswith("_seconds")}
            for model_id, m in metrics["models"].items()
        },
    }


def test_valid_table_passes() -> None:
    train.validate_syndrome_table(make_table())


def test_wrong_column_type_is_rejected() -> None:
    table = make_table()
    index = table.schema.get_field_index("sample_weight")
    table = table.set_column(index, "sample_weight", table["sample_weight"].cast(pa.int32()))
    with pytest.raises(train.InputContractError, match="schema"):
        train.validate_syndrome_table(table)


def test_missing_column_is_rejected() -> None:
    with pytest.raises(train.InputContractError, match="schema"):
        train.validate_syndrome_table(make_table().drop(["data_split"]))


@pytest.mark.parametrize(
    "rows, message",
    [
        ([*ROWS[:-1], ("te-2", 0.005, bytes(15), True, 25)], "16 binary values"),
        ([*ROWS[:-1], ("te-2", 0.005, bytes([2] + [0] * 15), True, 25)], "16 binary values"),
        ([*ROWS[:-1], ("te-2", 0.005, ZERO, True, 0)], "sample_weight must be positive"),
        ([*ROWS[:-1], ("te-1", 0.005, ZERO, True, 25)], "not unique"),
        ([row for row in ROWS if not row[0].startswith("va")], "'validation' is empty"),
    ],
)
def test_bad_rows_are_rejected(rows, message) -> None:
    with pytest.raises(train.InputContractError, match=message):
        train.validate_syndrome_table(make_table(rows))


def test_split_that_disagrees_with_fault_rate_is_rejected() -> None:
    with pytest.raises(train.InputContractError, match="does not match fault rate"):
        train.validate_syndrome_table(make_table(data_split="train"))


def test_features_keep_round_then_check_order() -> None:
    splits = train.syndrome_splits(make_table())
    row = splits["train"].example_id.index("tr-3")
    assert splits["train"].X[row].tolist() == list(_bits(2))
    assert train.SYNDROME_FEATURE_ORDER[2] == "round0_check2"
    assert train.SYNDROME_FEATURE_ORDER[4] == "round1_check0"
    assert len(train.SYNDROME_FEATURE_ORDER) == 16


def test_rows_stay_in_their_supplied_split() -> None:
    splits = train.syndrome_splits(make_table())
    assert splits["train"].example_id == ["tr-0", "tr-1", "tr-2", "tr-3", "tr-4", "tr-5"]
    assert splits["validation"].example_id == ["va-0", "va-1", "va-2"]
    assert splits["test"].example_id == ["te-0", "te-1", "te-2"]


def test_test_labels_do_not_influence_fitting_or_threshold() -> None:
    """This test checks that test labels never change the model or the threshold."""
    flipped = [(i, r, b, (not label) if i.startswith("te") else label, w) for i, r, b, label, w in ROWS]
    original = train.run_task_a(make_table())
    changed = train.run_task_a(make_table(flipped))
    keys = ("models", "threshold")
    assert {k: original.run_info[k] for k in keys} == {k: changed.run_info[k] for k in keys}
    assert original.metrics["models"] != changed.metrics["models"]


def test_prior_learns_the_weighted_error_share_of_train() -> None:
    result = train.run_task_a(make_table())
    learned = result.run_info["models"][train.PRIOR_MODEL_ID]["learned_error_prior"]
    # 130 error shots out of 1600 train shots
    assert learned == pytest.approx(130 / 1600)
    assert result.metrics["weighted_error_share"]["train"] == pytest.approx(130 / 1600)


def test_weight_counts_like_repeated_rows() -> None:
    """This test checks that a row with weight 2 fits like two rows of weight 1."""
    weighted = [("a", 0.0001, ZERO, False, 2), ("b", 0.0001, _bits(0), True, 3),
                ("c", 0.0001, _bits(0), False, 1)]
    repeated = [("a1", 0.0001, ZERO, False, 1), ("a2", 0.0001, ZERO, False, 1),
                ("b1", 0.0001, _bits(0), True, 1), ("b2", 0.0001, _bits(0), True, 1),
                ("b3", 0.0001, _bits(0), True, 1), ("c", 0.0001, _bits(0), False, 1)]
    models = []
    for rows in (weighted, repeated):
        data = train.syndrome_splits(make_table(rows))["train"]
        model = train.LogisticRegression(max_iter=1000, random_state=train.RANDOM_SEED)
        models.append(model.fit(data.X, data.y, sample_weight=data.weight))
    assert np.allclose(models[0].coef_, models[1].coef_, atol=1e-4)
    assert np.allclose(models[0].intercept_, models[1].intercept_, atol=1e-4)


def _split(labels, weights) -> train.SplitData:
    return train.SplitData(
        X=np.zeros((len(labels), 16), dtype=np.int8),
        y=np.array(labels, dtype=bool),
        weight=np.array(weights, dtype=np.int64),
        example_id=[str(i) for i in range(len(labels))],
    )


def test_weighted_metrics_match_hand_calculation() -> None:
    data = _split([True, False, False], [1, 3, 6])
    prediction = np.array([True, True, False])
    probability = np.array([0.5, 0.5, 0.0])
    metrics = train.evaluate(data, prediction, probability)
    assert metrics["logical_error_rate"] == pytest.approx(0.3)
    assert metrics["balanced_accuracy"] == pytest.approx((1 + 6 / 9) / 2)
    assert metrics["brier_score"] == pytest.approx(0.1)


def test_brier_is_none_without_probability() -> None:
    data = _split([True, False], [1, 1])
    assert train.evaluate(data, np.array([True, False]), None)["brier_score"] is None


def test_always_majority_has_balanced_accuracy_one_half() -> None:
    data = _split([True, False, False], [1, 50, 49])
    assert train.evaluate(data, np.zeros(3, dtype=bool), None)["balanced_accuracy"] == pytest.approx(0.5)


def test_threshold_minimizes_weighted_validation_error() -> None:
    data = _split([True, False, False], [5, 1, 100])
    probability = np.array([0.30, 0.20, 0.10])
    threshold, rate = train.choose_threshold(data, probability)
    assert threshold == 0.30
    assert rate == 0.0


def test_task_a_is_repeatable() -> None:
    first, second = train.run_task_a(make_table()), train.run_task_a(make_table())
    assert first.predictions == second.predictions
    assert _without_timings(first.metrics) == _without_timings(second.metrics)
    assert pickle.dumps(first.models[train.LOGISTIC_MODEL_ID]) == pickle.dumps(
        second.models[train.LOGISTIC_MODEL_ID]
    )


def _parquet_bytes(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_run_reads_only_the_ml_files_and_writes_results(monkeypatch, tmp_path) -> None:
    lake = {
        train.SYNDROME_ML_PATH: _parquet_bytes(make_table()),
        train.GOOGLE_ML_PATH: _parquet_bytes(make_google_table()),
    }
    read = []

    def fake_read(_settings, object_name):
        read.append(object_name)
        return lake[object_name]

    monkeypatch.setattr(train, "read_lake_bytes", fake_read)
    monkeypatch.setattr(train, "data_release", lambda: "test-release")
    monkeypatch.setattr(train, "code_revision", lambda: "abc123")

    train.run("run-1", results_dir=tmp_path)

    assert sorted(read) == sorted(lake)  # nothing but the two ML tables
    all_predictions = pq.read_table(tmp_path / "predictions.parquet")
    assert all_predictions.schema.names[:1] == ["example_id"]
    assert {"example_id", "model_id", "label", "prediction", "probability", "split"} <= set(
        all_predictions.schema.names
    )
    tasks = all_predictions["task"].to_pylist()
    predictions = all_predictions.filter(pa.array([task == train.TASK_A for task in tasks]))
    assert predictions.num_rows == 2 * len(ROWS)
    expected_split = {row[0]: train.syndrome_data_split(row[1]) for row in ROWS}
    for example_id, split in zip(predictions["example_id"].to_pylist(), predictions["split"].to_pylist()):
        assert split == expected_split[example_id]
    assert tasks.count(train.TASK_B) == 6 * make_google_table().num_rows
    # Task C's fixed subset is distance == 3 AND shot_index < 12_500; the toy
    # fixture's distance-3 shots are all 0..59, so every one of them qualifies
    # and gets exactly one model's prediction (the MLP, no baseline).
    distance_three_rows = sum(1 for r in make_google_table().to_pylist() if r["distance"] == 3)
    assert tasks.count(train.TASK_C) == distance_three_rows

    run_record = json.loads((tmp_path / "run.json").read_text())
    assert run_record["input_hashes"][train.GOOGLE_ML_PATH] == train.sha256_hex(lake[train.GOOGLE_ML_PATH])
    assert run_record["task_b_google"]["feature_order"] == train.GOOGLE_FEATURE_ORDER
    assert run_record["input_hashes"][train.SYNDROME_ML_PATH] == train.sha256_hex(lake[train.SYNDROME_ML_PATH])
    assert run_record["random_seed"] == train.RANDOM_SEED
    assert run_record["task_a_syndrome"]["feature_order"] == train.SYNDROME_FEATURE_ORDER
    assert run_record["task_a_syndrome"]["threshold"]["chosen_on"] == "validation"
    assert run_record["task_c_raw_detector"]["feature_order"] == train.RAW_BIT_FEATURE_ORDER
    assert run_record["task_c_raw_detector"]["threshold"]["chosen_on"] == "validation"
    assert set(run_record["model_files"]) == {
        f"models/{train.PRIOR_MODEL_ID}.pkl", f"models/{train.LOGISTIC_MODEL_ID}.pkl",
        f"models/{train.MLP_MODEL_ID}.pkl",
        *(f"models/task_b_d{d}_{name}.pkl" for d in (3, 5) for name in ("prior", "combined_logistic")),
    }

    all_metrics = json.loads((tmp_path / "metrics.json").read_text())
    blocks = [
        all_metrics["task_a_syndrome"],
        all_metrics["task_c_raw_detector"],
        *all_metrics["task_b_google"]["distances"].values(),
    ]
    for block in blocks:
        for m in block["models"].values():
            assert {"logical_error_rate", "balanced_accuracy", "brier_score",
                    "training_seconds", "prediction_seconds"} <= set(m)
    report = (tmp_path / "report.md").read_text()
    assert report.startswith("# Part II report")
    assert "## Task A" in report and "## Task B" in report and "## Task C" in report


DECODERS = train.GOOGLE_META_PREDICTION_COLUMNS


def _packed(fired: list[int], count: int) -> bytes:
    """This function packs fired detectors into a Stim b8 row."""
    data = bytearray((count + 7) // 8)
    for index in fired:
        data[index // 8] |= 1 << (index % 8)
    return bytes(data)


def _google_row(distance: int, shot_index: int) -> dict:
    """This function builds one toy Google shot from its distance and shot index."""
    count = train.GOOGLE_DETECTOR_COUNTS[distance]
    actual = shot_index % 3 == 0
    fired = list(range(shot_index % 7 + (5 if actual else 0)))
    predictions = {
        DECODERS[0]: actual if shot_index % 4 else not actual,
        DECODERS[1]: actual if shot_index % 5 else not actual,
        DECODERS[2]: actual if shot_index % 2 else not actual,
        DECODERS[3]: actual if shot_index % 6 else not actual,
    }
    experiment_id = f"surface_code_bX_d{distance}_r25_center_5_5"
    return {
        "example_id": f"{experiment_id}:{shot_index}",
        "experiment_id": experiment_id,
        "shot_index": shot_index,
        "distance": distance,
        "rounds": 25,
        "center_row": 5,
        "center_col": 5,
        "detector_count": count,
        "detector_event_count": len(fired),
        "detector_bits": _packed(fired, count),
        **predictions,
        "actual_observable_flip": actual,
        "data_split": train.google_data_split(shot_index),
    }


def make_google_table(shots: int = 60, rows: list[dict] | None = None) -> pa.Table:
    if rows is None:
        rows = [_google_row(d, s) for d in (3, 5) for s in range(shots)]
    return pa.Table.from_pylist(rows, schema=train.GOOGLE_INPUT_SCHEMA)


def _changed(**changes) -> pa.Table:
    """This function builds the toy Google table with shot 1 changed."""
    rows = make_google_table().to_pylist()
    rows[1].update(changes)
    return make_google_table(rows=rows)


def test_valid_google_table_passes() -> None:
    train.validate_google_table(make_google_table())


def test_toy_packing_is_little_endian_within_each_byte() -> None:
    packed = _packed([0, 9], 16)
    assert packed == bytes([0b0000_0001, 0b0000_0010])
    assert unpack_little_endian_bits(packed, 16)[:10] == (1, 0, 0, 0, 0, 0, 0, 0, 0, 1)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"detector_event_count": 99}, "detector_event_count does not equal"),
        ({"detector_bits": bytes(24)}, "expected ceil"),
        ({"detector_count": 600}, "does not match distance"),
        ({"data_split": "train"}, "does not match shot_index"),
        ({"example_id": "surface_code_bX_d3_r25_center_5_5:0"}, "not unique"),
    ],
)
def test_bad_google_rows_are_rejected(changes, message) -> None:
    with pytest.raises(train.InputContractError, match=message):
        train.validate_google_table(_changed(**changes))


def test_google_split_missing_for_one_distance_is_rejected() -> None:
    rows = [r for r in make_google_table().to_pylist()
            if not (r["distance"] == 5 and r["data_split"] == "validation")]
    with pytest.raises(train.InputContractError, match="distance 5: split 'validation' is empty"):
        train.validate_google_table(make_google_table(rows=rows))


def test_wrong_google_type_is_rejected() -> None:
    table = make_google_table()
    index = table.schema.get_field_index("shot_index")
    table = table.set_column(index, "shot_index", table["shot_index"].cast(pa.int32()))
    with pytest.raises(train.InputContractError, match="schema"):
        train.validate_google_table(table)


def test_zero_padding_bits_are_accepted(monkeypatch) -> None:
    """This test checks that a detector count that is not a multiple of 8 passes when its padding is zero."""
    # 12 detectors -> 2 bytes, the top 4 bits of the last byte are padding
    monkeypatch.setitem(train.GOOGLE_DETECTOR_COUNTS, 3, 12)
    table = make_google_table()
    assert len(table["detector_bits"][0].as_py()) == 2
    train.validate_google_table(table)


def test_nonzero_padding_bits_are_rejected(monkeypatch) -> None:
    """This test checks that a set padding bit is reported."""
    monkeypatch.setitem(train.GOOGLE_DETECTOR_COUNTS, 3, 12)
    rows = make_google_table().to_pylist()
    bits = bytearray(rows[1]["detector_bits"])
    bits[-1] |= 0b1000_0000
    rows[1]["detector_bits"] = bytes(bits)
    with pytest.raises(train.InputContractError, match="padding bits are not zero"):
        train.validate_google_table(make_google_table(rows=rows))


def test_old_model_files_are_removed(tmp_path) -> None:
    """This test checks that model files from an earlier run do not stay behind."""
    models_dir = tmp_path / train.MODELS_DIR_NAME
    models_dir.mkdir()
    (models_dir / "renamed_model.pkl").write_bytes(b"old")
    task_a = train.run_task_a(make_table())
    train.write_outputs(tmp_path, {train.TASK_A: task_a}, {})
    assert sorted(p.name for p in models_dir.iterdir()) == [
        f"{train.LOGISTIC_MODEL_ID}.pkl", f"{train.PRIOR_MODEL_ID}.pkl",
    ]


def test_distances_are_kept_separate_and_inputs_come_from_the_helper() -> None:
    table = make_google_table()
    for distance in (3, 5):
        splits = train.google_splits(table, distance)
        ids = [i for data in splits.values() for i in data.example_id]
        assert all(f"_d{distance}_" in example_id for example_id in ids)
        assert len(ids) == 60
        row = next(r for r in table.to_pylist() if r["example_id"] == splits["test"].example_id[0])
        assert tuple(splits["test"].X[0]) == train.google_meta_model_input(row)
        assert splits["test"].X.shape[1] == len(train.GOOGLE_FEATURE_ORDER) == 5


def test_all_task_b_models_use_identical_test_shots() -> None:
    result = train.run_task_b(make_google_table())
    predictions = pa.concat_tables(result.predictions).to_pylist()
    for distance in (3, 5):
        test_ids: dict[str, list[str]] = {}
        for row in predictions:
            if row["split"] == "test" and row["model_id"].startswith(f"task_b_d{distance}_"):
                test_ids.setdefault(row["model_id"], []).append(row["example_id"])
        assert len(test_ids) == 6  # prior, four supplied decoders, combined
        assert len({tuple(ids) for ids in test_ids.values()}) == 1


def test_supplied_decoder_metrics_match_hand_count() -> None:
    table = make_google_table()
    result = train.run_task_b(table)
    rows = [r for r in table.to_pylist() if r["distance"] == 3 and r["data_split"] == "test"]
    for column in DECODERS:
        wrong = sum(r[column] != r["actual_observable_flip"] for r in rows)
        m = result.metrics["distances"]["distance_3"]["models"][train.task_b_model_id(3, train.decoder_name(column))]
        assert m["logical_error_rate"] == pytest.approx(wrong / len(rows))
        assert m["brier_score"] is None and m["training_seconds"] is None


def test_task_b_test_labels_do_not_influence_fitting_or_threshold() -> None:
    rows = make_google_table().to_pylist()
    flipped = [{**r, "actual_observable_flip": not r["actual_observable_flip"]} if r["data_split"] == "test" else r
               for r in rows]
    original = train.run_task_b(make_google_table(rows=rows)).run_info["distances"]
    changed = train.run_task_b(make_google_table(rows=flipped)).run_info["distances"]
    for key in original:
        assert original[key]["models"] == changed[key]["models"]
        assert original[key]["threshold"] == changed[key]["threshold"]


def test_decoder_overlap_matches_hand_calculation() -> None:
    # 4 shots, all labels True
    X = np.array([
        [0.1, 1, 1, 1, 1],  # nobody wrong
        [0.1, 0, 1, 1, 1],  # only decoder 0 wrong
        [0.1, 0, 0, 1, 1],  # decoders 0 and 1 wrong
        [0.1, 0, 0, 0, 0],  # all wrong
    ])
    data = train.SplitData(X=X, y=np.ones(4, dtype=bool), weight=np.ones(4, dtype=np.int64),
                           example_id=["a", "b", "c", "d"])
    overlap = train.decoder_overlap(data, combined_prediction=np.array([True, True, True, False]))
    assert overlap["wrong_count_share"] == {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.0, "4": 0.25}
    assert overlap["all_wrong_share"] == 0.25
    assert overlap["disagree_share"] == 0.5
    names = [train.decoder_name(c) for c in DECODERS]
    pair = overlap["pairwise"][f"{names[0]} & {names[1]}"]
    assert pair["both_wrong_share"] == 0.5  # shots c and d
    assert pair["overlap"] == pytest.approx(2 / 3)
    assert overlap["combined_vs_best"] == {"fixed_share": 0.0, "broken_share": 0.0}


def test_task_b_is_repeatable() -> None:
    first, second = train.run_task_b(make_google_table()), train.run_task_b(make_google_table())
    assert first.predictions == second.predictions
    assert first.run_info == second.run_info


# ---------------------------------------------------------------------------
# Task C: bounded raw-detector prototype
# ---------------------------------------------------------------------------


def test_raw_detector_subset_keeps_only_distance_three_below_the_shot_limit() -> None:
    """This test checks that Task C's filter keeps distance 3, shot_index < 12_500 only."""
    limit = train.RAW_DETECTOR_SHOT_LIMIT
    rows = [
        _google_row(3, 10),            # kept
        _google_row(3, limit),         # dropped: at the limit, not below it
        _google_row(3, limit + 1),     # dropped: over the limit
        _google_row(5, 10),            # dropped: wrong distance
    ]
    table = make_google_table(rows=rows)
    splits = train.raw_detector_splits(table)
    ids = [i for data in splits.values() for i in data.example_id]
    assert ids == [rows[0]["example_id"]]


def test_unpack_detector_matrix_matches_the_single_row_helper() -> None:
    """This test checks that the vectorized Task C unpacker agrees, bit for
    bit, with the scalar course helper used for Task B's contract checks."""
    packed = [_packed([0, 9, 199], 200), _packed([], 200)]
    matrix = train.unpack_detector_matrix(packed)
    for row, bits in zip(matrix, packed):
        assert tuple(int(v) for v in row) == unpack_little_endian_bits(bits, 200)
    assert matrix.shape == (2, 200)


def test_unpack_detector_matrix_handles_an_empty_column() -> None:
    assert train.unpack_detector_matrix([]).shape == (0, train.RAW_DETECTOR_BIT_COUNT)


def test_raw_detector_rows_stay_in_their_supplied_split() -> None:
    """This test checks that Task C's split matches google_data_split(shot_index)
    for every row in the fixed subset, the same rule as Task B."""
    table = make_google_table()  # distances 3 and 5, shots 0..59
    splits = train.raw_detector_splits(table)
    for split, data in splits.items():
        for example_id in data.example_id:
            shot_index = int(example_id.rsplit(":", 1)[1])
            assert train.google_data_split(shot_index) == split
    assert all("_d3_" in example_id for data in splits.values() for example_id in data.example_id)


def test_raw_detector_features_are_200_bits_wide() -> None:
    splits = train.raw_detector_splits(make_google_table())
    assert len(train.RAW_BIT_FEATURE_ORDER) == 200
    for data in splits.values():
        assert data.X.shape[0] == len(data.example_id)
        if data.X.shape[0]:
            assert data.X.shape[1] == 200
            assert set(np.unique(data.X)) <= {0.0, 1.0}


def test_task_c_validates_its_input_table() -> None:
    """This test checks that run_task_c runs the same contract check as
    Task B before touching the data, so a broken Google table never reaches the MLP."""
    with pytest.raises(train.InputContractError, match="detector_event_count does not equal"):
        train.run_task_c(_changed(detector_event_count=99))


def test_task_c_test_labels_do_not_influence_fitting_or_threshold() -> None:
    """This test checks that flipping test-split labels never changes the
    fitted MLP or the chosen threshold, the same discipline as Tasks A and B."""
    rows = make_google_table().to_pylist()
    flipped = [{**r, "actual_observable_flip": not r["actual_observable_flip"]} if r["data_split"] == "test" else r
               for r in rows]
    original = train.run_task_c(make_google_table(rows=rows))
    changed = train.run_task_c(make_google_table(rows=flipped))
    keys = ("model", "threshold")
    assert {k: original.run_info[k] for k in keys} == {k: changed.run_info[k] for k in keys}
    assert original.metrics["models"] != changed.metrics["models"]


def test_task_c_is_repeatable() -> None:
    """This test checks that two fresh runs on the same input give the same
    predictions and run_info. MLPClassifier with a fixed random_state is
    deterministic for a given BLAS backend; a flake here points at
    multi-threaded BLAS non-determinism on the machine, not at the pipeline."""
    first, second = train.run_task_c(make_google_table()), train.run_task_c(make_google_table())
    assert first.predictions == second.predictions
    assert first.run_info == second.run_info