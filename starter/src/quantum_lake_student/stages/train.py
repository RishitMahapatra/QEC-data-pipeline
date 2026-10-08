"""Part II training stage (Tasks A, B and C).

This stage reads the two ML Parquet files from the lake, trains and evaluates the Task A, Task B and Task C models, and writes everything to results/part2/.
It never reads Gold, Silver or Bronze, and never writes to the lake.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import platform
import time
import uuid
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.neural_network import MLPClassifier

from quantum_lake_student import connections
from quantum_lake_student.config import Settings
from quantum_lake_student.ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    MODEL_SPLITS,
    google_data_split,
    google_meta_model_input,
    syndrome_data_split,
    syndrome_model_input,
    weighted_logical_error_rate,
)
from quantum_lake_student.models import StageResult, code_revision
from quantum_lake_student.stages import register_sources

#the two fixed lake object paths this stage is allowed to read — nothing else.
SYNDROME_ML_PATH = "ml/ml_syndrome_decoder_example.parquet" 
GOOGLE_ML_PATH = "ml/ml_google_decoder_example.parquet"

#this is the directory where everything gets written
RESULTS_PART2 = Path("results/part2")
MODELS_DIR_NAME = "models"

RANDOM_SEED = 4000 #a single fixed seed reused everywhere a model needs one

# Byte i of syndrome_bits is round i // 4, check i % 4.
SYNDROME_FEATURE_ORDER = [f"round{r}_check{c}" for r in range(4) for c in range(4)]

# Thresholds tried on validation: 0.01, 0.02, ..., 0.99. The fixed 0.01–0.99 grid (99 values) used everywhere a decision threshold is chosen on validation — defined once so Tasks A, B, C all search the identical grid.
THRESHOLD_GRID = [round(0.01 * step, 2) for step in range(1, 100)]

# Written from required-ml-tables.md, not imported from Part I, so the consumer checks the file against the contract on its own.
'''
The reason why we did that is because if Part II imported the schema straight from Part I's code, and someone later changed Part I by mistake, Part II would just go along with it without any warning. By typing the schema out again here by hand, Part II checks the actual file it received against its own copy of what it's supposed to look like. That's what lets it catch a mistake instead of blindly trusting Part I.
'''
#THIS IS THE SYNDROME INPUT SCHEMA 
SYNDROME_INPUT_SCHEMA = pa.schema([
    ("example_id", pa.string()), #identifier
    ("experiment_id", pa.string()), #identifier
    ("physical_fault_rate", pa.float64()), #determines which split a row belongs to
    ("syndrome_bits", pa.binary()), # the packed 16-bit binary blob
    #The round count and the blob count is always expected to be 4 for both cause its a 4X4= 16 bits
    ("round_count", pa.int32()), 
    ("check_count", pa.int32()),
    #THIS IS THE TARGET VARIABLE
    ("logical_error_label", pa.bool_()),#output is 0 or 1 -> boolean values
    ("sample_weight", pa.int64()), #this is the shot count a row represents
    ("data_split", pa.string()), #it is the pre-assigned train/validation/test tag from Part I
])

#THIS IS THE PREDICTION SCHEMA: it is the fixed output shape for every single prediction row this stage ever writes, across all three tasks and all models. One row = one model's prediction for one example in one split.
PREDICTIONS_SCHEMA = pa.schema([
    ("example_id", pa.string()), #identifier that traces back to the same input row -> every prediction retains the example_id 
    #The main purpose is to trace it across all the pipelines: from ML → Gold → Silver → Bronze
    #task and model_id let predictions.parquet hold Task A, B, and C's outputs all mixed together in one file, distinguishable by these two columns
    ("task", pa.string()),
    ("model_id", pa.string()),
    
    ("label", pa.bool_()),#this is the true label / answer
    ("prediction", pa.bool_()), #this is the model prediction 
    ("probability", pa.float64()), #this can be nullable as the google decoders only give yes or no instead of any specific number
    ("split", pa.string()), #This split means whether the particular data came from test, validation or train split. This will help the user to categorise of extract only the test split.
])

#The three variables below are just string constants naming Task A and its two models, used consistently instead of retyping literal strings everywhere
TASK_A = "task_a_syndrome"
PRIOR_MODEL_ID = "task_a_weighted_prior"
LOGISTIC_MODEL_ID = "task_a_weighted_logistic_regression"

#GOOGLE INPUT SCHEMA FOR GOOGLE DATASETS: Includes the hand types schema description for the ml_google_decoder_example table
GOOGLE_INPUT_SCHEMA = pa.schema([
    ("example_id", pa.string()), #identifier for traceability
    ("experiment_id", pa.string()), #identifier 
    ("shot_index", pa.int64()), #identifies the shot and also decides its split via google_data_split
    ("distance", pa.int32()),#the chip size -> only 3 or 5
    ("rounds", pa.int32()), 
    
    #locate the experiment on the chip
    ("center_row", pa.int32()),
    ("center_col", pa.int32()),

    #packed detector readings 
    ("detector_count", pa.int32()),
    ("detector_event_count", pa.int32()),
    ("detector_bits", pa.binary()),
    #this unpacks the four supplied decoders' yes/no predictions (e.g. pymatching_prediction, etc.) into the schema in one line, instead of hardcoding four separate lines.
    *((name, pa.bool_()) for name in GOOGLE_META_PREDICTION_COLUMNS),
    ("actual_observable_flip", pa.bool_()),
    ("data_split", pa.string()),
])

GOOGLE_DETECTOR_COUNTS = {3: 200, 5: 600} #for distance 3 -> 200 bits and for distance 5-> 600 bits 

GOOGLE_FEATURE_ORDER = ["detector_event_density", *GOOGLE_META_PREDICTION_COLUMNS] 

TASK_B = "task_b_google" #the string id naming Task B, same pattern as TASK_A before.


def task_b_model_id(distance: int, name: str) -> str:
    """This function builds a Task B model id, e.g. task_b_d3_pymatching.
    Task B runs separately for distance 3 and distance 5, and each distance has its own prior baseline, its own four supplied decoders, and its own combined model — so every model needs a unique id that also says which distance it belongs to. This builds that id consistently, e.g. task_b_d3_prior, task_b_d5_combined_logistic. Without this, you'd risk typo'd or clashing ids scattered across the code."""
    return f"task_b_d{distance}_{name}"


def decoder_name(column: str) -> str:
    """This function turns 'pymatching_prediction' into 'pymatching'.
    The four supplied decoder columns in the input table are named like pymatching_prediction, union_find_prediction, etc. (ending in _prediction). This strips that suffix to get the plain decoder name (pymatching), which is then fed into task_b_model_id to build that decoder's own model id (task_b_d3_pymatching). It's a small, single-purpose string transform used wherever the code needs to go from "column name" to "human-readable decoder name." """
    return column.removesuffix("_prediction")

# Custom error type so "bad input table" is distinguishable from other
# ValueErrors when caught, instead of using generic ValueError everywhere.
class InputContractError(ValueError):
    """Raised when an ML input table does not match its contract."""


'''This function will read one file's raw bytes from wherever the lake lives.
a local folder (dev/testing) or MinIO/S3 (real deployment). try/finally always closes the network 
connection even if read() fails.'''
def read_lake_bytes(settings: Settings, object_name: str) -> bytes:
    """This function reads one file from the lake (MinIO or a local folder)."""
    if settings.lake_backend == "local":
        return (settings.local_lake_root / object_name).read_bytes()
    response = connections.minio_client(settings).get_object(settings.s3_bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def sha256_hex(data: bytes) -> str:
    """This function returns the SHA-256 hash of some bytes."""
    return hashlib.sha256(data).hexdigest()


def parquet_table(data: bytes) -> pa.Table:
    """This function turns Parquet bytes into a table."""
    return pq.read_table(io.BytesIO(data))


def validate_syndrome_table(table: pa.Table) -> None:
    """This function checks the syndrome table against its contract and stops the run if it fails.

    It checks columns and types, nulls, unique ids, that all splits exist,
    16 binary bits per row, 4 rounds and 4 checks, positive weights, and that
    each row's split matches its fault rate.
    """
    errors: list[str] = [] #The errors that are encountered are collected as a list instead of stopping the execution whenever there is a problem faced. This way the error msg will report several issues

    #Compares the table's actual column names+types against SYNDROME_INPUT_SCHEMA.
    actual = [(f.name, f.type) for f in table.schema.remove_metadata()] #this is the actual  column names
    expected = [(f.name, f.type) for f in SYNDROME_INPUT_SCHEMA] #this is the expected column names
    if actual != expected: #if it does not match 
        raise InputContractError(f"ml_syndrome_decoder_example: schema {actual} != contract {expected}") #fatal error 

    for name in table.column_names: #this statement checks every columns for nulls -> as pyarrow won't stop a null from getting in -> manually loop in order to check for null counts
        if table[name].null_count: 
            errors.append(f"{name} has {table[name].null_count} null values") #appending the errors in case of null 

    ids = table["example_id"].to_pylist() 
    if len(set(ids)) != len(ids):
        errors.append(f"example_id is not unique ({len(ids) - len(set(ids))} duplicates)")

    splits = table["data_split"].to_pylist() #this checks the split: train or test or validation. if there is missing rows for either one of the splits, then it will skip entirely a pipeline stage. 
    for split in MODEL_SPLITS:
        if split not in splits: 
            errors.append(f"split {split!r} is empty") #errors logged in whenever one of the above mentioned split is missing

    rows = zip( # Zips together the columns needed to check each row individually.
        ids,
        table["syndrome_bits"].to_pylist(),
        table["round_count"].to_pylist(),
        table["check_count"].to_pylist(),
        table["sample_weight"].to_pylist(),
        table["physical_fault_rate"].to_pylist(),
        splits,
    )
    for example_id, bits, rounds, checks, weight, rate, split in rows:
        try:
            syndrome_model_input(bits)
        except (TypeError, ValueError) as exc:
            errors.append(f"{example_id}: {exc}")
        if rounds != 4 or checks != 4: #this checks if every row has 4 X4 = 16 bits
            errors.append(f"{example_id}: round_count/check_count must be 4/4")
        if weight is None or weight <= 0: # Weight (shot count) must be a real positive number, never zero/negative/missing.
            errors.append(f"{example_id}: sample_weight must be positive")
        if rate is not None and split != syndrome_data_split(rate): # Recomputes which split this fault rate SHOULD be in, and checks the row's stored split agrees — catches a mislabeled row.
            errors.append(f"{example_id}: data_split {split!r} does not match fault rate {rate}")

    _raise_if_errors("ml_syndrome_decoder_example", errors)


def _raise_if_errors(table_name: str, errors: list[str]) -> None:
    """This function raises one error listing the first few problems, if there are any."""
    if errors:
        shown = "; ".join(errors[:5]) + (f"; ... {len(errors) - 5} more" if len(errors) > 5 else "")
        raise InputContractError(f"{table_name}: {shown}")


def validate_google_table(table: pa.Table) -> None:
    """This function checks the Google table against its contract and stops the run if it fails.

    It checks columns and types, nulls, unique ids and shots, that each row's
    split matches its shot index, 200/600 detectors per distance, all splits
    per distance, packed byte length and zero padding, that the event count
    equals the fired bits, and that the predictions are binary.
    """
    errors: list[str] = []

    actual = [(f.name, f.type) for f in table.schema.remove_metadata()]
    expected = [(f.name, f.type) for f in GOOGLE_INPUT_SCHEMA]
    if actual != expected:
        raise InputContractError(f"ml_google_decoder_example: schema {actual} != contract {expected}")

    for name in table.column_names:
        if table[name].null_count:
            errors.append(f"{name} has {table[name].null_count} null values")
    if errors:
        _raise_if_errors("ml_google_decoder_example", errors)

    ids = table["example_id"].to_pylist()
    if len(set(ids)) != len(ids):
        errors.append(f"example_id is not unique ({len(ids) - len(set(ids))} duplicates)")
    shots = list(zip(table["experiment_id"].to_pylist(), table["shot_index"].to_pylist()))
    if len(set(shots)) != len(shots):
        errors.append("(experiment_id, shot_index) is not unique")

    distances = table["distance"].to_pylist()
    splits = table["data_split"].to_pylist()
    for distance in sorted(set(distances)):
        if distance not in GOOGLE_DETECTOR_COUNTS:
            errors.append(f"unexpected distance {distance}")
        present = {split for d, split in zip(distances, splits) if d == distance}
        for split in MODEL_SPLITS:
            if split not in present:
                errors.append(f"distance {distance}: split {split!r} is empty")

    columns = ["example_id", "shot_index", "distance", "detector_count", "detector_event_count",
               "detector_bits", "data_split", *GOOGLE_META_PREDICTION_COLUMNS]
    for row in table.select(columns).to_pylist():
        example_id, count, bits = row["example_id"], row["detector_count"], row["detector_bits"]
        if row["data_split"] != google_data_split(row["shot_index"]):
            errors.append(f"{example_id}: data_split does not match shot_index {row['shot_index']}")
        if GOOGLE_DETECTOR_COUNTS.get(row["distance"]) != count:
            errors.append(f"{example_id}: detector_count {count} does not match distance {row['distance']}")
        if len(bits) != (count + 7) // 8:
            errors.append(f"{example_id}: {len(bits)} detector bytes, expected ceil({count}/8)")
            continue
        # Bits are little-endian per byte, so padding sits in the high bits of the last byte.
        if count % 8 and bits[-1] >> (count % 8):
            errors.append(f"{example_id}: padding bits are not zero")
        if int.from_bytes(bits, "little").bit_count() != row["detector_event_count"]:
            errors.append(f"{example_id}: detector_event_count does not equal the set detector bits")
        try:
            google_meta_model_input(row)
        except (TypeError, ValueError) as exc:
            errors.append(f"{example_id}: {exc}")

    _raise_if_errors("ml_google_decoder_example", errors)


@dataclass
class SplitData:
    """All rows of one split as aligned arrays: inputs X, labels y, weights and example ids."""

    X: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    example_id: list[str]

    @property
    def total_weight(self) -> int:
        return int(self.weight.sum())


def syndrome_splits(table: pa.Table) -> dict[str, SplitData]:
    """This function turns the syndrome table into train/validation/test arrays.

    The 16 inputs come from the course helper syndrome_model_input, and rows
    stay in the split Part I gave them.
    """
    rows: dict[str, dict[str, list]] = {
        split: {"X": [], "y": [], "weight": [], "example_id": []} for split in MODEL_SPLITS
    }
    columns = zip(
        table["example_id"].to_pylist(),
        table["syndrome_bits"].to_pylist(),
        table["logical_error_label"].to_pylist(),
        table["sample_weight"].to_pylist(),
        table["data_split"].to_pylist(),
    )
    for example_id, bits, label, weight, split in columns:
        group = rows[split]
        group["X"].append(syndrome_model_input(bits))
        group["y"].append(bool(label))
        group["weight"].append(int(weight))
        group["example_id"].append(example_id)

    return {
        split: SplitData(
            X=np.array(group["X"], dtype=np.int8).reshape(-1, len(SYNDROME_FEATURE_ORDER)),
            y=np.array(group["y"], dtype=bool),
            weight=np.array(group["weight"], dtype=np.int64),
            example_id=group["example_id"],
        )
        for split, group in rows.items()
    }


def google_splits(table: pa.Table, distance: int) -> dict[str, SplitData]:
    """This function turns the Google shots of one distance into train/validation/test arrays.

    The 5 inputs (event density + the 4 decoder answers) come from the course
    helper google_meta_model_input. Every shot gets weight 1.
    """
    rows = table.filter(pc.equal(table["distance"], distance))
    columns = ["example_id", "detector_count", "detector_event_count",
               *GOOGLE_META_PREDICTION_COLUMNS, "actual_observable_flip", "data_split"]
    grouped: dict[str, dict[str, list]] = {
        split: {"X": [], "y": [], "example_id": []} for split in MODEL_SPLITS
    }
    for row in rows.select(columns).to_pylist():
        group = grouped[row["data_split"]]
        group["X"].append(google_meta_model_input(row))
        group["y"].append(bool(row["actual_observable_flip"]))
        group["example_id"].append(row["example_id"])
    return {
        split: SplitData(
            X=np.array(group["X"], dtype=np.float64).reshape(-1, len(GOOGLE_FEATURE_ORDER)),
            y=np.array(group["y"], dtype=bool),
            weight=np.ones(len(group["y"]), dtype=np.int64),
            example_id=group["example_id"],
        )
        for split, group in grouped.items()
    }


def weighted_error_share(data: SplitData) -> float:
    """This function returns the weighted share of rows with a positive label."""
    return float(data.weight[data.y].sum() / data.weight.sum())


def evaluate(
    data: SplitData,
    prediction: np.ndarray,
    probability: np.ndarray | None,
) -> dict[str, float | None]:
    """This function computes the weighted error rate, balanced accuracy and Brier score.

    The Brier score is None when the model gives no probability.
    """
    return {
        "logical_error_rate": weighted_logical_error_rate(data.y, prediction, data.weight),
        "balanced_accuracy": float(
            balanced_accuracy_score(data.y, prediction, sample_weight=data.weight)
        ),
        "brier_score": None
        if probability is None
        else float(brier_score_loss(data.y, probability, sample_weight=data.weight)),
    }


def choose_threshold(validation: SplitData, probability: np.ndarray) -> tuple[float, float]:
    """This function picks the threshold with the lowest weighted error rate on validation.

    Ties go to the threshold closest to 0.5. It returns the threshold and its
    validation error rate.
    """
    best: tuple[float, float, float] | None = None
    for threshold in THRESHOLD_GRID:
        rate = weighted_logical_error_rate(validation.y, probability >= threshold, validation.weight)
        candidate = (rate, abs(threshold - 0.5), threshold)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[2], best[0]


@dataclass
class TaskResult:
    """What one task adds to the results: predictions, metrics, run details and models."""

    predictions: list[pa.Table] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    run_info: dict = field(default_factory=dict)
    models: dict[str, object] = field(default_factory=dict)


def prediction_table(
    task: str, model_id: str, split: str, data: SplitData, prediction, probability
) -> pa.Table:
    """This function builds the predictions.parquet rows for one model on one split."""
    rows = len(data.example_id)
    return pa.Table.from_arrays(
        [
            pa.array(data.example_id, pa.string()),
            pa.array([task] * rows, pa.string()),
            pa.array([model_id] * rows, pa.string()),
            pa.array(data.y, pa.bool_()),
            pa.array(np.asarray(prediction, dtype=bool), pa.bool_()),
            pa.nulls(rows, pa.float64()) if probability is None
            else pa.array(np.asarray(probability, dtype=np.float64), pa.float64()),
            pa.array([split] * rows, pa.string()),
        ],
        schema=PREDICTIONS_SCHEMA,
    )


def run_task_a(table: pa.Table) -> TaskResult:
    """This function runs Task A: a weighted prior and a weighted logistic regression on the syndromes.

    Both models are fitted on train with sample_weight, the threshold is
    chosen on validation, and the metrics are computed on test.
    """
    validate_syndrome_table(table)
    splits = syndrome_splits(table)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    result = TaskResult()

    prior = DummyClassifier(strategy="prior")
    started = time.perf_counter()
    prior.fit(train.X, train.y, sample_weight=train.weight)
    prior_train_seconds = time.perf_counter() - started

    logistic = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    started = time.perf_counter()
    logistic.fit(train.X, train.y, sample_weight=train.weight)
    logistic_train_seconds = time.perf_counter() - started

    error_column = list(logistic.classes_).index(True)
    validation_probability = logistic.predict_proba(validation.X)[:, error_column]
    threshold, validation_rate_at_threshold = choose_threshold(validation, validation_probability)

    # Predictions are saved for every split; metrics only come from test.
    prior_error_column = list(prior.classes_).index(True)
    timings: dict[str, dict[str, float]] = {}
    for model_id, model in ((PRIOR_MODEL_ID, prior), (LOGISTIC_MODEL_ID, logistic)):
        for split in MODEL_SPLITS:
            data = splits[split]
            started = time.perf_counter()
            if model is prior:
                probability = prior.predict_proba(data.X)[:, prior_error_column]
                prediction = prior.predict(data.X).astype(bool)
            else:
                probability = logistic.predict_proba(data.X)[:, error_column]
                prediction = probability >= threshold
            seconds = time.perf_counter() - started
            if split == "test":
                timings[model_id] = {"prediction_seconds": seconds}
                result.metrics[model_id] = {
                    **evaluate(test, prediction, probability),
                    "training_seconds": prior_train_seconds if model is prior else logistic_train_seconds,
                    "prediction_seconds": seconds,
                }
            result.predictions.append(
                prediction_table(TASK_A, model_id, split, data, prediction, probability)
            )
        timings[model_id]["training_seconds"] = result.metrics[model_id]["training_seconds"]

    result.metrics = {
        "evaluated_split": "test",
        "test_rows": len(test.example_id),
        "test_physical_shots": test.total_weight,
        "weighted_error_share": {split: weighted_error_share(splits[split]) for split in MODEL_SPLITS},
        "models": result.metrics,
    }
    result.models = {PRIOR_MODEL_ID: prior, LOGISTIC_MODEL_ID: logistic}
    result.run_info = {
        "input_table": SYNDROME_ML_PATH,
        "target": "logical_error_label",
        "feature_helper": "quantum_lake_student.ml.syndrome_model_input",
        "feature_order": SYNDROME_FEATURE_ORDER,
        "split_rule": "data_split column from Part I (syndrome_data_split): "
        "validation = fault rate 0.0005, test = 0.005, train = the other five",
        "split_rows": {split: len(splits[split].example_id) for split in MODEL_SPLITS},
        "split_physical_shots": {split: splits[split].total_weight for split in MODEL_SPLITS},
        "weights": "sample_weight used in fitting both models and in every metric",
        "models": {
            PRIOR_MODEL_ID: {"class": "sklearn.dummy.DummyClassifier", "params": {"strategy": "prior"},
                             "learned_error_prior": float(prior.class_prior_[prior_error_column])},
            LOGISTIC_MODEL_ID: {
                "class": "sklearn.linear_model.LogisticRegression",
                "params": {"max_iter": 1000, "random_state": RANDOM_SEED},
                "iterations": int(logistic.n_iter_[0]),
                "intercept": float(logistic.intercept_[0]),
                "coefficients": dict(zip(SYNDROME_FEATURE_ORDER, map(float, logistic.coef_[0]))),
            },
        },
        "threshold": {
            "value": threshold,
            "chosen_on": "validation",
            "rule": "lowest weighted logical-error rate over 0.01..0.99, tie -> closest to 0.5",
            "validation_logical_error_rate": validation_rate_at_threshold,
        },
        "timings_seconds": timings,
    }
    return result


def decoder_overlap(test: SplitData, combined_prediction: np.ndarray) -> dict:
    """This function measures how much the four supplied decoders make the same mistakes on test.

    It returns how often 0-4 decoders are wrong together, the shared mistakes
    of every pair, and what the combined model fixes or breaks compared with
    the best single decoder.
    """
    names = [decoder_name(column) for column in GOOGLE_META_PREDICTION_COLUMNS]
    errors = {name: test.X[:, 1 + k].astype(bool) != test.y for k, name in enumerate(names)}
    wrong_count = np.sum(list(errors.values()), axis=0)
    shots = len(test.y)

    pairwise = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            both = np.sum(errors[first] & errors[second])
            either = np.sum(errors[first] | errors[second])
            pairwise[f"{first} & {second}"] = {
                "both_wrong_share": float(both / shots),
                "overlap": float(both / either) if either else 1.0,
            }

    best = min(names, key=lambda name: (errors[name].mean(), name))
    combined_wrong = combined_prediction != test.y
    return {
        "wrong_count_share": {str(k): float(np.mean(wrong_count == k)) for k in range(len(names) + 1)},
        "all_wrong_share": float(np.mean(wrong_count == len(names))),
        "disagree_share": float(np.mean((wrong_count > 0) & (wrong_count < len(names)))),
        "pairwise": pairwise,
        "best_single_decoder": best,
        "combined_vs_best": {
            "fixed_share": float(np.mean(errors[best] & ~combined_wrong)),
            "broken_share": float(np.mean(~errors[best] & combined_wrong)),
        },
    }


def run_task_b(table: pa.Table) -> TaskResult:
    """This function runs Task B separately for each distance.

    It scores a prior, the four supplied decoders and one combined logistic
    regression on the same test shots, and measures decoder overlap.
    """
    validate_google_table(table)
    result = TaskResult()
    metrics: dict[str, dict] = {}
    run_info: dict[str, dict] = {}

    for distance in sorted(set(table["distance"].to_pylist())):
        splits = google_splits(table, distance)
        train, validation, test = splits["train"], splits["validation"], splits["test"]
        prior_id = task_b_model_id(distance, "prior")
        combined_id = task_b_model_id(distance, "combined_logistic")

        prior = DummyClassifier(strategy="prior")
        started = time.perf_counter()
        prior.fit(train.X, train.y)
        prior_train_seconds = time.perf_counter() - started
        prior_column = list(prior.classes_).index(True)

        combined = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        started = time.perf_counter()
        combined.fit(train.X, train.y)
        combined_train_seconds = time.perf_counter() - started
        flip_column = list(combined.classes_).index(True)
        threshold, validation_rate = choose_threshold(
            validation, combined.predict_proba(validation.X)[:, flip_column]
        )

        models: dict[str, dict] = {}
        combined_test_prediction = None
        for split in MODEL_SPLITS:
            data = splits[split]

            started = time.perf_counter()
            probability = prior.predict_proba(data.X)[:, prior_column]
            prediction = prior.predict(data.X).astype(bool)
            prior_seconds = time.perf_counter() - started
            result.predictions.append(prediction_table(TASK_B, prior_id, split, data, prediction, probability))
            if split == "test":
                models[prior_id] = {**evaluate(data, prediction, probability),
                                    "training_seconds": prior_train_seconds,
                                    "prediction_seconds": prior_seconds}

            # Supplied decoders are already computed, so they have no probability or timings.
            for k, column in enumerate(GOOGLE_META_PREDICTION_COLUMNS):
                model_id = task_b_model_id(distance, decoder_name(column))
                prediction = data.X[:, 1 + k].astype(bool)
                result.predictions.append(prediction_table(TASK_B, model_id, split, data, prediction, None))
                if split == "test":
                    models[model_id] = {**evaluate(data, prediction, None),
                                        "training_seconds": None, "prediction_seconds": None}

            started = time.perf_counter()
            probability = combined.predict_proba(data.X)[:, flip_column]
            prediction = probability >= threshold
            combined_seconds = time.perf_counter() - started
            result.predictions.append(prediction_table(TASK_B, combined_id, split, data, prediction, probability))
            if split == "test":
                combined_test_prediction = prediction
                models[combined_id] = {**evaluate(data, prediction, probability),
                                       "training_seconds": combined_train_seconds,
                                       "prediction_seconds": combined_seconds}

        assert combined_test_prediction is not None
        key = f"distance_{distance}"
        metrics[key] = {
            "evaluated_split": "test",
            "test_shots": len(test.y),
            "flip_share": {split: weighted_error_share(splits[split]) for split in MODEL_SPLITS},
            "models": models,
            "decoder_overlap": decoder_overlap(test, combined_test_prediction),
        }
        run_info[key] = {
            "detector_count": GOOGLE_DETECTOR_COUNTS[distance],
            "split_rows": {split: len(splits[split].y) for split in MODEL_SPLITS},
            "models": {
                prior_id: {"class": "sklearn.dummy.DummyClassifier", "params": {"strategy": "prior"},
                           "learned_flip_prior": float(prior.class_prior_[prior_column])},
                combined_id: {
                    "class": "sklearn.linear_model.LogisticRegression",
                    "params": {"max_iter": 1000, "random_state": RANDOM_SEED},
                    "iterations": int(combined.n_iter_[0]),
                    "intercept": float(combined.intercept_[0]),
                    "coefficients": dict(zip(GOOGLE_FEATURE_ORDER, map(float, combined.coef_[0]))),
                },
            },
            "threshold": {
                "value": threshold,
                "chosen_on": "validation",
                "rule": "lowest logical-error rate over 0.01..0.99, tie -> closest to 0.5",
                "validation_logical_error_rate": validation_rate,
            },
        }
        result.models[prior_id] = prior
        result.models[combined_id] = combined

    result.metrics = {"distances": metrics}
    result.run_info = {
        "input_table": GOOGLE_ML_PATH,
        "target": "actual_observable_flip",
        "feature_helper": "quantum_lake_student.ml.google_meta_model_input",
        "feature_order": GOOGLE_FEATURE_ORDER,
        "split_rule": "data_split column from Part I (google_data_split): odd shot_index = test, "
        "shot_index % 10 == 8 = validation, other even = train",
        "distances_kept_separate": True,
        "weights": "none: every row is one hardware shot",
        "distances": run_info,
    }
    return result


# ---------------------------------------------------------------------------
# Task C: bounded raw-detector prototype (part-2-ai-ml.md Task C, rubric #8)
# ---------------------------------------------------------------------------

TASK_C = "task_c_raw_detector"
MLP_MODEL_ID = "task_c_raw_detector_mlp"

# The fixed, bounded subset every team uses: distance 3 only, first 12,500
# shots of each of the four distance-three experiments.
RAW_DETECTOR_DISTANCE = 3
RAW_DETECTOR_SHOT_LIMIT = 12_500
RAW_DETECTOR_BIT_COUNT = GOOGLE_DETECTOR_COUNTS[RAW_DETECTOR_DISTANCE]  # 200

RAW_BIT_FEATURE_ORDER = [f"detector_bit_{i}" for i in range(RAW_DETECTOR_BIT_COUNT)]

# Small, fixed architecture -- this is a pipeline-consumer check, not an
# architecture search (part-2-ai-ml.md: "You are not required to design or
# research a more complex model").
MLP_HIDDEN_LAYER_SIZES = (32,)
MLP_MAX_ITER = 300


def unpack_detector_matrix(detector_bits_column: list[bytes]) -> np.ndarray:
    """Unpack one BYTEA-per-row detector_bits column into a (rows, 200) 0/1 matrix.

    Same bit convention as ml.unpack_little_endian_bits and the Gold design
    doc's Section 6.5 (numpy.unpackbits(..., bitorder="little")): little-endian
    within each byte, sequential across bytes. Distance 3 has exactly 200 bits
    in exactly 25 bytes (200 = 25 * 8), so there is no padding to drop here --
    the slice below is defensive, not load-bearing for this distance.
    """
    if not detector_bits_column:
        return np.zeros((0, RAW_DETECTOR_BIT_COUNT), dtype=np.float64)
    raw = np.frombuffer(b"".join(detector_bits_column), dtype=np.uint8)
    raw = raw.reshape(len(detector_bits_column), -1)
    bits = np.unpackbits(raw, axis=1, bitorder="little")
    return bits[:, :RAW_DETECTOR_BIT_COUNT].astype(np.float64)


def raw_detector_splits(table: pa.Table) -> dict[str, SplitData]:
    """Build train/validation/test arrays for Task C's fixed subset.

    Filters to distance == 3 AND shot_index < 12_500 first, then uses the
    supplied data_split column unchanged (already verified against
    google_data_split by validate_google_table).
    """
    subset = table.filter(
        pc.and_(
            pc.equal(table["distance"], RAW_DETECTOR_DISTANCE),
            pc.less(table["shot_index"], RAW_DETECTOR_SHOT_LIMIT),
        )
    )
    grouped: dict[str, dict[str, list]] = {
        split: {"bits": [], "y": [], "example_id": []} for split in MODEL_SPLITS
    }
    columns = subset.select(
        ["example_id", "detector_bits", "actual_observable_flip", "data_split"]
    ).to_pylist()
    for row in columns:
        group = grouped[row["data_split"]]
        group["bits"].append(bytes(row["detector_bits"]))
        group["y"].append(bool(row["actual_observable_flip"]))
        group["example_id"].append(row["example_id"])

    return {
        split: SplitData(
            X=unpack_detector_matrix(group["bits"]),
            y=np.array(group["y"], dtype=bool),
            weight=np.ones(len(group["y"]), dtype=np.int64),
            example_id=group["example_id"],
        )
        for split, group in grouped.items()
    }


def run_task_c(table: pa.Table) -> TaskResult:
    """Run Task C: one small MLP on unpacked raw detector bits.

    Fixed subset: distance == 3 AND shot_index < 12_500, across the four
    distance-three experiments. Threshold is chosen on validation only, same
    discipline as Tasks A and B; metrics are computed on test only. No
    baseline is required for this task (part-2-ai-ml.md).
    """
    validate_google_table(table)
    splits = raw_detector_splits(table)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    result = TaskResult()

    mlp = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_SEED,
    )
    started = time.perf_counter()
    mlp.fit(train.X, train.y)
    train_seconds = time.perf_counter() - started

    flip_column = list(mlp.classes_).index(True)
    threshold, validation_rate = choose_threshold(
        validation, mlp.predict_proba(validation.X)[:, flip_column]
    )

    model_metrics: dict[str, dict] = {}
    train_fit: dict = {}
    for split in MODEL_SPLITS:
        data = splits[split]
        started = time.perf_counter()
        probability = mlp.predict_proba(data.X)[:, flip_column]
        prediction = probability >= threshold
        seconds = time.perf_counter() - started
        result.predictions.append(
            prediction_table(TASK_C, MLP_MODEL_ID, split, data, prediction, probability)
        )
        if split == "train":
            # Only reported next to the test numbers to show over-fitting; no choice is made on it.
            train_metrics = evaluate(data, prediction, probability)
            train_fit = {
                "logical_error_rate": train_metrics["logical_error_rate"],
                "brier_score": train_metrics["brier_score"],
            }
        if split == "test":
            model_metrics[MLP_MODEL_ID] = {
                **evaluate(data, prediction, probability),
                "training_seconds": train_seconds,
                "prediction_seconds": seconds,
            }

    result.metrics = {
        "evaluated_split": "test",
        "test_shots": len(test.example_id),
        "flip_share": {split: weighted_error_share(splits[split]) for split in MODEL_SPLITS},
        "models": model_metrics,
        "train_fit": train_fit,
    }
    result.models = {MLP_MODEL_ID: mlp}
    result.run_info = {
        "input_table": GOOGLE_ML_PATH,
        "target": "actual_observable_flip",
        "feature_helper": "quantum_lake_student.stages.train.unpack_detector_matrix "
        "(numpy.unpackbits, bitorder='little'; no course-supplied helper covers this task)",
        "feature_order": RAW_BIT_FEATURE_ORDER,
        "subset_filter": f"distance == {RAW_DETECTOR_DISTANCE} AND shot_index < {RAW_DETECTOR_SHOT_LIMIT}",
        "split_rule": "data_split column from Part I (google_data_split), same rule as Task B, "
        "restricted to the fixed subset above",
        "split_rows": {split: len(splits[split].example_id) for split in MODEL_SPLITS},
        "model": {
            "class": "sklearn.neural_network.MLPClassifier",
            "params": {
                "hidden_layer_sizes": MLP_HIDDEN_LAYER_SIZES,
                "max_iter": MLP_MAX_ITER,
                "random_state": RANDOM_SEED,
            },
            "n_iter": int(mlp.n_iter_),
        },
        "threshold": {
            "value": threshold,
            "chosen_on": "validation",
            "rule": "lowest logical-error rate over 0.01..0.99, tie -> closest to 0.5",
            "validation_logical_error_rate": validation_rate,
        },
    }
    return result


def dependency_versions() -> dict[str, str]:
    """This function returns the installed versions of the libraries that affect the results."""
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn", "pyarrow"):
        versions[package] = metadata.version(package)
    return versions


def data_release() -> str:
    """This function returns the course data-release name from the bundle manifest."""
    manifest = register_sources._load_manifest()
    return f"{manifest['release_name']} v{manifest['bundle_version']} ({manifest['release_date']})"


def _fmt(value: float | None, digits: int = 4) -> str:
    """This function formats a number for the report, or 'n/a' when it is missing."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def task_a_report(task_a: TaskResult) -> str:
    """This function writes the Task A section of report.md from this run's numbers."""
    metrics, info = task_a.metrics, task_a.run_info
    prior_m = metrics["models"][PRIOR_MODEL_ID]
    logistic_m = metrics["models"][LOGISTIC_MODEL_ID]
    share = metrics["weighted_error_share"]
    threshold = info["threshold"]
    beats = logistic_m["logical_error_rate"] < prior_m["logical_error_rate"]
    rows = "\n".join(
        f"| `{model_id}` | {_fmt(m['logical_error_rate'])} | {_fmt(m['balanced_accuracy'])} | "
        f"{_fmt(m['brier_score'])} | {m['training_seconds']:.4f} | {m['prediction_seconds']:.4f} |"
        for model_id, m in metrics["models"].items()
    )
    return f"""## Task A: weighted syndrome decoder

**Input and target.** `{info['input_table']}` ({sum(info['split_rows'].values()):,} rows,
{sum(info['split_physical_shots'].values()):,} physical shots). Each row is one distinct
(experiment, 16-bit syndrome, label) combination; `sample_weight` says how many
simulated shots it stands for. The input is the 16 syndrome bits in round-then-check
order from `syndrome_model_input`; the target is `logical_error_label`.

**Splits.** The supplied `data_split` is used unchanged:

| Split | Fault rate(s) | Rows | Physical shots | Weighted error share |
| --- | --- | ---: | ---: | ---: |
| train | 0.00001, 0.00005, 0.0001, 0.001, 0.01 | {info['split_rows']['train']:,} | {info['split_physical_shots']['train']:,} | {share['train']:.4f} |
| validation | 0.0005 | {info['split_rows']['validation']:,} | {info['split_physical_shots']['validation']:,} | {share['validation']:.4f} |
| test | 0.005 | {info['split_rows']['test']:,} | {info['split_physical_shots']['test']:,} | {share['test']:.4f} |

**Models.** Both are fitted on train with `sample_weight`, so a row counts as
many times as the shots it represents and the 70M observations are never
expanded.
- *Weighted prior*: ignores the syndrome and always predicts the majority label
  (no logical error), with the weighted train error share as its probability.
  It is the bar any decoder must beat, because logical errors are rare.
- *Weighted logistic regression*: one weight per syndrome bit. It suits this table
  because the input is a short, fixed-length binary vector and the model gives a
  probability, so a Brier score can be reported. The decision threshold
  ({threshold['value']}) was chosen on validation only ({threshold['rule']}).

**Test results** (fault rate 0.005, every metric weighted by `sample_weight`):

| Model | Logical-error rate | Balanced accuracy | Brier score | Train s | Predict s |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

**Interpretation.** Logical errors are rare and their share differs a lot between
the splits ({share['train']:.1%} train, {share['validation']:.1%} validation,
{share['test']:.1%} test). The prior therefore has a low error rate but a balanced
accuracy of exactly 0.5: it never detects an error. The logistic regression
{"has a lower" if beats else "does not have a lower"} logical-error rate than the
prior on test ({_fmt(logistic_m['logical_error_rate'])} vs {_fmt(prior_m['logical_error_rate'])}),
and a balanced accuracy of {_fmt(logistic_m['balanced_accuracy'])}. Its Brier score
({_fmt(logistic_m['brier_score'])} vs {_fmt(prior_m['brier_score'])} for the prior)
shows whether its probabilities are more informative than the constant prior.
The threshold was tuned on a split where only {share['validation']:.1%} of shots fail,
which favors predicting "no error"; the test split fails about
{share['test'] / share['validation']:.0f}x as often, so a threshold that is good on
validation need not be good on test. We report this as it is instead of tuning on test.

**Discarded information and limitations.**
- A linear model adds up bit effects independently. It cannot represent
  combinations such as "these two checks fired in consecutive rounds", which is
  how errors show up in a syndrome.
- The physical fault rate is not an input (it defines the split), so the model
  cannot adapt to how noisy the experiment was.
- Identical syndromes appear with both labels, so no decoder using only these
  16 bits can be correct on all shots.
- One test fault rate is a single operating point; results at other noise levels
  may differ.
"""


def task_b_report(task_b: TaskResult) -> str:
    """This function writes the Task B section of report.md from this run's numbers."""
    info = task_b.run_info
    sections = []
    for key, block in task_b.metrics["distances"].items():
        distance = int(key.removeprefix("distance_"))
        details = info["distances"][key]
        overlap = block["decoder_overlap"]
        models = block["models"]
        combined_id = task_b_model_id(distance, "combined_logistic")
        best_id = task_b_model_id(distance, overlap["best_single_decoder"])
        combined_rate = models[combined_id]["logical_error_rate"]
        best_rate = models[best_id]["logical_error_rate"]
        coefficients = details["models"][combined_id]["coefficients"]
        rows = "\n".join(
            f"| `{model_id}` | {_fmt(m['logical_error_rate'])} | {_fmt(m['balanced_accuracy'])} | "
            f"{_fmt(m['brier_score'])} | {_fmt(m['training_seconds'])} | {_fmt(m['prediction_seconds'])} |"
            for model_id, m in models.items()
        )
        shares = overlap["wrong_count_share"]
        pairs = "\n".join(
            f"| {pair} | {v['both_wrong_share']:.4f} | {v['overlap']:.4f} |"
            for pair, v in overlap["pairwise"].items()
        )
        sections.append(f"""### Distance {distance}

{details['split_rows']['train']:,} train / {details['split_rows']['validation']:,} validation /
{block['test_shots']:,} test shots; {block['flip_share']['test']:.1%} of test shots have a
logical flip. Combined-decoder threshold {details['threshold']['value']} (chosen on validation).

| Model | Logical-error rate | Balanced accuracy | Brier score | Train s | Predict s |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

Supplied decoders give only yes/no answers that were computed by the data
provider, so their Brier score and timings are not applicable.

**Do the supplied decoders make the same mistakes?** Share of test shots where
0, 1, 2, 3 or 4 of them are wrong: {", ".join(f"{k}: {v:.1%}" for k, v in shares.items())}.
They disagree on {overlap['disagree_share']:.1%} of shots; on {overlap['all_wrong_share']:.1%}
all four are wrong, and no combination of their answers can fix those.

| Pair | Both wrong | Overlap (both / either wrong) |
| --- | ---: | ---: |
{pairs}

**Effect on the combined model.** The best single decoder is
`{overlap['best_single_decoder']}` ({_fmt(best_rate)}). The combined model
{"improves on it" if combined_rate < best_rate else "does not improve on it"}
({_fmt(combined_rate)}): it fixes {overlap['combined_vs_best']['fixed_share']:.1%} of test shots
that decoder gets wrong and breaks {overlap['combined_vs_best']['broken_share']:.1%} it got right.
Learned coefficients: {", ".join(f"{name} {value:+.2f}" for name, value in coefficients.items())}.
A large coefficient means the model relies on that input.
""")

    return f"""## Task B: supplied and combined Google decoders

**Input and target.** `{info['input_table']}`: one row per real hardware shot,
target `actual_observable_flip`. Distance 3 and distance 5 are evaluated
completely separately (different detector counts, different experiments).
The combined decoder sees exactly five inputs from `google_meta_model_input`:
{", ".join(f"`{name}`" for name in info['feature_order'])}. The split is the supplied
shot-based `data_split` ({info['split_rule']}); every model is scored on the same test shots.

**Models.** A prior baseline (always the most common outcome in train); the
four supplied decoder predictions as given; and one logistic regression that
learns how much to trust each decoder and the shot's event density. A linear
combination suits this table because the inputs are a handful of aligned
decoder votes plus one noise summary, and it gives a probability.

{chr(10).join(sections)}
**Discarded information and limitations.**
- The combined model never sees the raw detector bits, only their density, so
  it cannot tell where on the chip or in which round the events happened.
- A linear model weighs each vote independently; it cannot learn rules such as
  "trust decoder X only when the shot is very noisy".
- Distance 5 comes from a single experiment (one chip location), so its results
  may not generalize to other locations.
- Each shot covers 25 rounds, so the logical-flip share is close to 50% and all
  decoders are far from perfect on this data.
"""


def task_c_report(task_c: TaskResult) -> str:
    """Write the Task C section of report.md from this run's numbers."""
    metrics, info = task_c.metrics, task_c.run_info
    m = metrics["models"][MLP_MODEL_ID]
    share = metrics["flip_share"]
    fit = metrics["train_fit"]
    threshold = info["threshold"]
    return f"""## Task C: bounded raw-detector prototype

**Input and target.** `{info['input_table']}`, filtered to `{info['subset_filter']}`
({sum(info['split_rows'].values()):,} rows across the four distance-three experiments).
Each row's input is its own 200 raw detector bits, unpacked bit-for-bit from the
packed `detector_bits` column (little-endian within each byte, matching how Part I
stored them); the target is `actual_observable_flip`, the same target as Task B.

**Splits.** The supplied shot-based `data_split` is used unchanged, restricted to
this subset:

| Split | Rows | Flip share |
| --- | ---: | ---: |
| train | {info['split_rows']['train']:,} | {share['train']:.1%} |
| validation | {info['split_rows']['validation']:,} | {share['validation']:.1%} |
| test | {info['split_rows']['test']:,} | {share['test']:.1%} |

**Model.** A small multilayer perceptron
(`hidden_layer_sizes={info['model']['params']['hidden_layer_sizes']}`) trained
directly on the 200 raw bits, with no feature engineering. This is a
pipeline-consumer check, not an architecture search: the point is that the
prepared data can feed a model expecting raw per-detector inputs, not that it
beats Task B's combined decoder. The decision threshold ({threshold['value']})
was chosen on validation only ({threshold['rule']}).

**Test results:**

| Model | Logical-error rate | Balanced accuracy | Brier score | Train s | Predict s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `{MLP_MODEL_ID}` | {_fmt(m['logical_error_rate'])} | {_fmt(m['balanced_accuracy'])} | {_fmt(m['brier_score'])} | {m['training_seconds']:.4f} | {m['prediction_seconds']:.4f} |

**Interpretation and limitations.** A logical-error rate of
{_fmt(m['logical_error_rate'])} on test, where {share['test']:.1%} of shots
flip, means the MLP does no better than guessing. Its Brier score
({_fmt(m['brier_score'])}) is worse than the 0.25 of always predicting 50%, so
its probabilities are confidently wrong. On its own training rows it reaches a
logical-error rate of {_fmt(fit['logical_error_rate'])} and a Brier score of
{_fmt(fit['brier_score'])}: it has memorised the
{info['split_rows']['train']:,} training shots instead of learning a rule that
carries over to new shots. The training loop also stops at
`max_iter={info['model']['params']['max_iter']}`, so it may end before fully
converging. We keep this result as it is: the task is a bounded
pipeline-consumer check, and tuning the model is outside its scope. The
supplied decoders, which use the layout of the code, do clearly better on
distance-three test shots (Task B).

**What the flat 200-bit vector hides.** Each bit's position in the vector is
arbitrary to the model: it has no notion of which detector is physically
adjacent to which other detector, or which QEC round a bit belongs to. The real
error signal in this data is spatial (neighbouring detectors firing together
marks a local error) and temporal (a detector firing in consecutive rounds means
something different from firing once); flattening to 200 independent inputs
discards both before the MLP ever sees the data. This caps what a small MLP on
this representation can learn, regardless of how it is tuned -- which is exactly
why the assignment treats this as a bounded prototype, not a competition.
"""


def write_report(path: Path, tasks: dict[str, TaskResult]) -> None:
    """This function writes report.md with a section for every task that ran."""
    sections = [
        "# Part II report\n\nGenerated by `make train`; all numbers below come from "
        "`metrics.json` and\n`run.json` of the same run.\n"
    ]
    if TASK_A in tasks:
        sections.append(task_a_report(tasks[TASK_A]))
    if TASK_B in tasks:
        sections.append(task_b_report(tasks[TASK_B]))
    if TASK_C in tasks:
        sections.append(task_c_report(tasks[TASK_C]))
    path.write_text("\n".join(sections))


def write_outputs(results_dir: Path, tasks: dict[str, TaskResult], run_record: dict) -> None:
    """This function writes the model files, predictions.parquet, metrics.json, run.json and report.md."""
    models_dir = results_dir / MODELS_DIR_NAME
    models_dir.mkdir(parents=True, exist_ok=True)
    # Remove model files from earlier runs, so a renamed or dropped model leaves nothing behind.
    for old_model in models_dir.glob("*.pkl"):
        old_model.unlink()

    model_hashes = {}
    for task in tasks.values():
        for model_id, model in task.models.items():
            data = pickle.dumps(model)
            (models_dir / f"{model_id}.pkl").write_bytes(data)
            model_hashes[f"{MODELS_DIR_NAME}/{model_id}.pkl"] = sha256_hex(data)
    run_record["model_files"] = model_hashes

    pieces = [piece for task in tasks.values() for piece in task.predictions]
    pq.write_table(pa.concat_tables(pieces), results_dir / "predictions.parquet")
    metrics = {name: task.metrics for name, task in tasks.items()}
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (results_dir / "run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True))
    write_report(results_dir / "report.md", tasks)


def run(model_run_id: str | None = None, results_dir: Path = RESULTS_PART2) -> StageResult:
    """This function runs Part II end to end and rewrites results/part2."""
    model_run_id = model_run_id or str(uuid.uuid4())
    settings = Settings.from_environment()
    result = StageResult(stage="train", run_id=model_run_id)

    syndrome_bytes = read_lake_bytes(settings, SYNDROME_ML_PATH)
    google_bytes = read_lake_bytes(settings, GOOGLE_ML_PATH)
    syndrome_table = parquet_table(syndrome_bytes)
    google_table = parquet_table(google_bytes)

    tasks = {
        TASK_A: run_task_a(syndrome_table),
        TASK_B: run_task_b(google_table),
        TASK_C: run_task_c(google_table),
    }

    result.input_count = syndrome_table.num_rows + google_table.num_rows
    result.output_count = sum(piece.num_rows for task in tasks.values() for piece in task.predictions)
    result.finish()

    run_record = {
        "run_id": model_run_id,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "code_revision": code_revision(),
        "data_release": data_release(),
        "input_hashes": {
            SYNDROME_ML_PATH: sha256_hex(syndrome_bytes),
            GOOGLE_ML_PATH: sha256_hex(google_bytes),
        },
        "random_seed": RANDOM_SEED,
        "dependencies": dependency_versions(),
        **{name: task.run_info for name, task in tasks.items()},
    }
    write_outputs(results_dir, tasks, run_record)
    return result