# PART 2 REPORT

# Task A design decisions — weighted syndrome decoder

## What Task A is

Task A asks for a decoder that predicts `logical_error_label` from the
simulated syndrome table, `ml/ml_syndrome_decoder_example.parquet`. The
spec (`part-2-ai-ml.md`) fixes the scope:

- The input is the **ordered 4-by-4 syndrome**: 16 binary values, round
  first and then check, taken from the course helper
  `syndrome_model_input`.
- **Exactly two comparisons**: a weighted majority/prior baseline and one
  weighted linear classifier.
- One row is not one shot. Each of the 75,598 rows stands for `sample_weight`
  identical simulated shots, 70 million in total, and the spec forbids
  expanding them into individual rows.
- The split is supplied and fixed by physical fault rate: validation is
  `0.0005`, test is `0.005`, train is the other five files.
- Physical weights must be used in fitting where supported and in **all**
  reported metrics.

## How we decided to implement it

**Why Part II validates the table again.** Part I already checks the table
when it exports it, but the spec says to "validate both input-table schemas
before training". `validate_syndrome_table()` therefore checks the file from
the consumer side: exact columns and types, no nulls, unique `example_id`,
all three splits present, 16 binary values per row, 4 rounds and 4 checks,
positive weights, and that each row's `data_split` matches
`syndrome_data_split` for its fault rate. If anything fails, the run stops
before any model is fitted.

**Why logistic regression is the one linear model.** The spec allows
exactly one weighted linear classifier and names logistic regression as an
example. We picked it because it accepts `sample_weight`, it gives a
probability (so a Brier score can be reported, not just yes/no metrics),
and its 16 coefficients can be read directly as "how much does this check
in this round push toward a logical error".

**Why the weights are used everywhere and rows are never expanded.** A row
with weight 9,987,291 has to count as that many shots, otherwise a rare
pattern would matter as much as the most common one. Both models are fitted
with `sample_weight`, and `evaluate()` passes the same weights to the
logical-error rate, balanced accuracy and Brier score. A test checks that a
row with weight 2 fits exactly like two rows with weight 1, which is the
property that makes expanding the 70 million shots unnecessary.

**Why the baseline is a weighted prior.** `DummyClassifier(strategy=
"prior")`, fitted with the weights, learns only the weighted share of
logical errors in train (4.3%). It always predicts "no error" and reports
that share as its probability. Logical errors are rare, so this is already
a strong baseline on error rate, and it gives the linear model a fair bar
on both the yes/no metrics and the Brier score.

**Why the threshold is chosen on validation by error rate.** The logistic
regression gives a probability, so one choice remains: where to cut it into
yes/no. `choose_threshold()` tries 0.01 to 0.99 on validation only and keeps
the value with the lowest weighted logical-error rate, since that is the
task's main metric. Ties go to the value closest to 0.5, so the choice is
always the same. Test is never used for this.

**Why we report a result that does not beat the baseline.** On test the
logistic regression has a logical-error rate of 0.1041 against 0.1040 for
the prior, with a better Brier score (0.0910 against 0.0969). The reason is
the threshold: validation has only 1.1% errors, so the chosen threshold is
0.93, which almost never predicts an error, while test has 10.4% errors.
We report this as it is instead of tuning on test. The spec says a model
that does not beat its baseline is a complete result when it is explained.

**Why predictions are saved for every split.** All metrics come from test
only, but `predictions.parquet` keeps train and validation rows as well,
each with its `example_id`. That way any syndrome example in the ML table
can be traced from its prediction back to Gold, Silver and Bronze.

**Why the run is repeatable.** The split is fixed, nothing is shuffled, the
default `lbfgs` solver is deterministic and `RANDOM_SEED` is fixed on top of
that. Two runs give byte-identical predictions and model files; only the
timings differ.

## Rubric comparison (item 6 — Weighted syndrome model, 12 points)

| Rubric requirement (10–12 point band) | How Task A meets it |
| --- | --- |
| Correct supplied partitions and physical weights | `syndrome_splits()` groups rows by the existing `data_split` column, which `validate_syndrome_table()` has already checked against `syndrome_data_split`; `sample_weight` is passed to both `.fit()` calls and to every metric in `evaluate()` |
| Weighted prior and one weighted linear model | `DummyClassifier(strategy="prior")` and one `LogisticRegression`, both fitted on train with `sample_weight` in `run_task_a()` |
| Required metrics and timings | Weighted logical-error rate, balanced accuracy and Brier score from `evaluate()`; `.fit()` and prediction on test are timed, stored in `metrics["models"]` |
| Clear, data-based interpretation of imbalance and the result | `task_a_report()` states the error share of every split (4.3% / 1.1% / 10.4%), explains why the prior scores 0.5 balanced accuracy, and why a threshold chosen on validation does not carry over to test |

Nothing here depends on the linear model winning — rubric item 6 grades
the partitions, the weights, the baseline and the honesty of the
interpretation, not the score.

## Related checklist items this satisfies

- "The weighted majority/prior and one weighted linear classifier run." —
  both are fitted and scored in `run_task_a()`.
- "Supplied partitions are used without moving examples between them." —
  `test_rows_stay_in_their_supplied_split` and
  `test_split_that_disagrees_with_fault_rate_is_rejected`.
- "Physical weights are used in all Task A evaluation metrics and in fitting
  where supported." — `test_weight_counts_like_repeated_rows`,
  `test_prior_learns_the_weighted_error_share_of_train` and
  `test_weighted_metrics_match_hand_calculation`.
- "Tests cover required columns/types, split use ... weighting,
  repeatability, and metric calculations" — also
  `test_wrong_column_type_is_rejected`,
  `test_test_labels_do_not_influence_fitting_or_threshold` and
  `test_task_a_is_repeatable`.

# Task B design decisions — supplied and combined Google decoders

## What Task B is

Task B works on the real hardware table,
`ml/ml_google_decoder_example.parquet`, where every row is one shot. The
spec (`part-2-ai-ml.md`) asks, for **distance three and distance five
separately**:

- a majority/prior baseline;
- an evaluation of each of the **four supplied decoder predictions** on the
  same test shots;
- one simple linear combined decoder with exactly **five inputs**:
  normalized detector-event density and the four supplied predictions;
- any threshold chosen on validation only;
- a short description of whether the supplied decoders make the same or
  different mistakes, and how that affects the combined model.

The target is `actual_observable_flip`. The split is supplied per shot:
odd `shot_index` is test, even with `shot_index % 10 == 8` is validation,
the rest is train.

## How we decided to implement it

**Why the two distances never share anything.** Distance 3 has 200
detectors and four experiments (200,000 shots); distance 5 has 600
detectors and one experiment (50,000 shots). `google_splits()` filters the
table to one distance before anything else, and `run_task_b()` loops over
the distances, so each one gets its own prior, its own combined model, its
own threshold and its own metrics. No row of one distance can reach a model
of the other.

**Why the Google table gets extra checks.** On top of the shared checks,
`validate_google_table()` verifies what the contract names for packed
rows: `ceil(detector_count / 8)` bytes, zero padding bits,
`detector_event_count` equal to the number of fired bits, 200 or 600
detectors for the distance, and a unique `(experiment_id, shot_index)`. The
supplied data has no padding at all, because 200 and 600 are both multiples
of 8, so the padding check cannot fail on it. It is still there because the
contract requires it, and it is tested with a made-up count of 12 detectors.

**Why the supplied decoders are read, not run.** Their predictions were
computed by the data provider and sit on the same row as the label. Scoring
them means comparing two columns of that row, which also guarantees that
every model is judged on identical test shots. They only give yes/no
answers, so their Brier score and their training and prediction times are
recorded as not applicable instead of being invented.

**Why the five inputs come only from the helper.** `google_meta_model_input`
returns the event density (fired detectors divided by all detectors) and
the four predictions as 0/1. We use exactly that vector and build nothing
else. Columns 1 to 4 of the input are therefore the supplied decoders'
answers themselves, aligned with the label by construction.

**Why the combined decoder is a logistic regression.** The task asks for one
simple linear model. Logistic regression learns one coefficient per input,
which reads as "how much to trust this decoder", and it gives a probability.
Its threshold is chosen with the same `choose_threshold()` as Task A, on
validation only (0.51 for distance 3, 0.55 for distance 5).

**Why every shot has weight 1.** A row here is one real shot, not an
aggregate, so there is no physical weight. Giving each row weight 1 lets
Task B reuse the same `evaluate()` and `choose_threshold()` as Task A, so
the metrics of the two tasks are computed by the same code.

**Why overlap is measured per shot and per pair.** The spec asks whether
the decoders make the same mistakes. `decoder_overlap()` counts, on test,
how many of the four are wrong on each shot, how often each pair is wrong
together, and which shots the combined model fixes or breaks compared with
the best single decoder. In most pairs the decoders share less than half of
their mistakes; two distance-3 pairs share slightly more (53%). All four
are wrong together on 14.4% of distance-3 shots and 10.9% of distance-5
shots, and no combination of their answers can fix those.

**Why we report that the combination only ties the best decoder.** On test,
the four supplied decoders rank the same way at both distances (logical-error
rate, d=3 / d=5): tensor network contraction 0.3967 / 0.3952, belief matching
0.4013 / 0.3994, correlated matching 0.4220 / 0.4226, PyMatching
0.4364 / 0.4548. This matches Part I query 2. They disagree on 55.4% (d=3)
and 64.8% (d=5) of test shots, so their mistakes are partly different. The
combined model still reaches only 0.3977 at both distances: compared with
tensor network contraction it fixes about as many shots as it breaks (d=3:
6.6% fixed, 6.7% broken; d=5: 5.3% fixed, 5.6% broken). Among the decoders,
its largest coefficient is on tensor network contraction (0.53 / 0.59) and
its smallest on PyMatching (0.17 / 0.22), so it mostly follows the best
decoder. The disagreement alone is not enough: a fixed linear vote cannot
tell which decoder is right on a given shot.

## Rubric comparison (item 7 — Google combined decoder, 12 points)

| Rubric requirement (10–12 point band) | How Task B meets it |
| --- | --- |
| Distance-three and distance-five work is separate | `google_splits(table, distance)` filters to one distance first; `run_task_b()` fits, thresholds and scores each distance on its own, and `metrics.json` has one block per distance |
| All four supplied decoders are compared on identical test shots | Each decoder's prediction is read from the same row as the label; `test_all_task_b_models_use_identical_test_shots` checks that all six models share the same test `example_id`s |
| The five-feature linear combined decoder is aligned correctly | Inputs come only from `google_meta_model_input`; `GOOGLE_FEATURE_ORDER` is recorded in `run.json`; one `LogisticRegression` per distance |
| Validation-only choices and decoder-error overlap are explained honestly | Threshold from `choose_threshold()` on validation; `decoder_overlap()` feeds the overlap tables in `task_b_report()`, which states that the combined model does not beat the best single decoder |

Nothing here needs the combined decoder to win — rubric item 7 grades the
separation, the alignment, the identical test shots and the honest
explanation of the overlap.

## Related checklist items this satisfies

- "Distance-three and distance-five decoder comparisons are separate." —
  `test_distances_are_kept_separate_and_inputs_come_from_the_helper`.
- "All four supplied decoders are compared on the same held-out shots." —
  `test_all_task_b_models_use_identical_test_shots` and
  `test_supplied_decoder_metrics_match_hand_count`.
- "The linear combined decoder uses event density and the four aligned
  predictions." — the inputs are exactly what `google_meta_model_input`
  returns, enforced in `google_splits()`.
- "The short report explains decoder-error overlap ..." — this half of the
  checklist line is Task B's job: `decoder_overlap()`, tested by
  `test_decoder_overlap_matches_hand_calculation`.
- "Tests cover ... bit order, event counts ..." —
  `test_toy_packing_is_little_endian_within_each_byte`,
  `test_bad_google_rows_are_rejected`,
  `test_zero_padding_bits_are_accepted` and
  `test_nonzero_padding_bits_are_rejected`.

# Task C design decisions — bounded raw-detector prototype

## What Task C is

Task C asks for one small multilayer perceptron (MLP) trained directly on
**unpacked raw detector bits**, not on any derived feature. The spec
(`part-2-ai-ml.md`) fixes the scope deliberately narrow:

- Only **distance-three** rows.
- Only **`shot_index < 12_500`** per experiment (so four distance-three
  experiments × 12,500 shots = a fixed, bounded subset, not the full table).
- Each row's input is its **200 raw detector bits** (distance 3 has exactly
  200 detectors), unpacked bit-for-bit — no event counts, no decoder votes,
  no engineered features.
- Target is `actual_observable_flip`, same target as Task B.
- No required baseline (unlike Tasks A and B).
- In this part we focus on making a **pipeline-consumer check**: can the prepared data
  feed a model that expects raw per-detector inputs, not "build the best
  decoder."

## How we decided to implement it

**Why an MLP for this table.** The spec asks for one small MLP, and it is
also the simplest model that suits this input. Each row is 200 raw detector
bits with no engineered features, so the model has to find useful
combinations of bits by itself. Whether the logical observable flipped
depends on *which* detectors fire together, not on each detector alone: the
same detector firing can mean opposite things depending on its neighbours.
A linear model, like the logistic regressions of Tasks A and B, gives each
input one fixed weight and adds them up, so it cannot represent such
combinations. One hidden layer of 32 units can, because each unit learns its
own weighted mix of the 200 bits before the output combines them. The MLP
also takes the flat 200-bit vector directly, which is exactly what this task
tests: that the prepared ML table can feed a model expecting raw
per-detector inputs.

**Why a fixed small MLP, not a search.** Since the specs explicitly mention not to focus on the best architecture, we picked one small, fixed architecture (`hidden_layer_sizes=(32,)`,
`max_iter=300`) up front and did not tune it. Tuning would spend effort
the rubric doesn't reward and risks implicitly turning this into a
competition the spec says to avoid. Because of this cap, the MLP can stop before fully converging, which is expected since the cap keeps the run fast and repeatable.

**Why we built our own bit-unpacking helper.** Tasks A and B use
course-supplied helpers (`syndrome_model_input`, `google_meta_model_input`)
to turn packed bytes into model inputs. No course helper does this for
*raw* detector bits at the per-bit level Task C needs, so we wrote
`unpack_detector_matrix()` ourselves, using `numpy.unpackbits(...,
bitorder="little")` — the same little-endian-per-byte convention Part I
used to pack the bits in the first place (documented in part1.md,
section 6.5). We reused the existing convention rather than inventing
a new one, so Task C's inputs stay consistent with how the rest of the
pipeline reads these bytes.

**Why we filter before unpacking, not after.** `raw_detector_splits()`
filters the table to `distance == 3 AND shot_index < 12_500` first, then
groups by the *existing* `data_split` column. We don't recompute splits
ourselves, the split was already assigned by Part I (`google_data_split`)
and already verified against that rule by `validate_google_table()`. Task
C's job is to consume that split, restricted to its bounded subset, not to
invent a new split rule.

**Why no baseline model.** The spec's "Required procedure" (global rule
across all tasks) says to validate schemas, use validation for thresholds,
and test only for final metrics — but it does **not** list a required
baseline for Task C the way it does for Tasks A and B. We followed the
spec literally: one model, no baseline, for this task only.

**Why the threshold is still chosen on validation.** Even though Task C's
own section doesn't repeat it, the "Required procedure" section applies
globally ("use validation data for choices and thresholds" is listed once,
for all tasks). So `run_task_c()` reuses the exact same `choose_threshold()`
grid search (0.01–0.99, ties toward 0.5) that Tasks A and B use, applied to
validation only, metrics computed on test only — same discipline, not a
separate decision.

**Why training/prediction time is recorded per split.** The checklist
("Part II: results and tests") requires training time and prediction time
to be reported wherever applicable. `run_task_c()` times `mlp.fit()` once
and times `predict_proba()` separately for every split, same pattern as
Tasks A and B, so the timing numbers are directly comparable across tasks.

**Why the report explicitly discusses what the flat vector hides.** The
spec requires, in the student's own words, an explanation of what a flat
200-bit vector does not show (detector position, neighborhood, change over
QEC rounds) and why that limits the prototype. We treat this as a required
deliverable, not an optional aside — `task_c_report()` always includes it
automatically.

**Why we report a result that is no better than guessing.** On test the
MLP has a logical-error rate of 0.5056, while 50.0% of the test shots flip,
so it does no better than a coin flip. Its Brier score (0.4026) is worse
than the 0.25 of always predicting 50%, so its probabilities are
confidently wrong. On its own training rows it reaches 0.147 (Brier 0.106):
it has memorised the 20,000 training shots instead of learning a rule that
carries over to new shots. The 300-iteration cap also means it may stop before fully converging, which is why the tests print a
`ConvergenceWarning`. We keep this result as it is: tuning is outside the scope of this project.

## Rubric comparison (item 8)

| Rubric requirement (5–6 point band) | How Task C meets it |
| --- | --- |
| The required bounded distance-three MLP runs on the fixed subset | `RAW_DETECTOR_DISTANCE = 3`, `RAW_DETECTOR_SHOT_LIMIT = 12_500`; `raw_detector_splits()` filters to exactly `distance == 3 AND shot_index < 12_500` before anything else happens |
| Bit order and split are correct | `unpack_detector_matrix()` uses `bitorder="little"`, matching Part I's packing convention; split comes from the already-validated `data_split` column, not recomputed |
| Training/prediction time is recorded | `run_task_c()` times `.fit()` once and `.predict_proba()` per split, stored in `metrics["models"][MLP_MODEL_ID]` |
| The report clearly explains information hidden by a flat input | `task_c_report()`'s closing paragraph names spatial adjacency and temporal (across-round) structure as what gets discarded, and explains why that caps a flat-vector model regardless of tuning |

Nothing here claims or needs a specific accuracy number, rubric item 8,
like the rest of Part II, does not grade model performance, only that the
prototype is correctly scoped, correctly built, and honestly explained.

## Related checklist items this satisfies

- "The raw-detector MLP uses only distance-three rows with
  `shot_index < 12_500` and the supplied split." — enforced in
  `raw_detector_splits()`.
- "The short report explains decoder-error overlap and what a flat raw-bit
  input does not show explicitly." — the decoder-overlap half is Task B's
  job (`decoder_overlap()`); the flat-raw-bit half is Task C's job
  (`task_c_report()`'s closing section), so between the two reports this
  checklist line is fully covered.
- "Tests cover ... bit order and event counts ... repeatability ... metric
  calculations" — covered by `test_unpack_detector_matrix_matches_the_
  single_row_helper`, `test_raw_detector_features_are_200_bits_wide`,
  `test_task_c_is_repeatable`, and the shared `evaluate()`/`choose_threshold()`
  tests Tasks A/B/C all exercise.


# Repeatability, evaluation and tests

## Rubric comparison (item 9)

| Rubric requirement (9–10 point band) | How Part II meets it |
| --- | --- |
| One command regenerates model files, predictions, metrics, and run facts | `make train` (`python -m quantum_lake_student.cli train`) reads only the two ML Parquet tables and rewrites `results/part2/`: the 7 model files under `models/`, `predictions.parquet`, `metrics.json`, `run.json` and `report.md`. `test_run_reads_only_the_ml_files_and_writes_results` checks that nothing else is read |
| Input hashes, settings, feature order, splits, seeds, and timings are recorded | `run.json` records the data release, the SHA-256 of both ML tables (`input_hashes`), the code revision, the dependency versions (Python, numpy, pyarrow, scikit-learn, scipy), `random_seed`, the model files, and per task the input table, target, feature helper, `feature_order`, `split_rule`, split sizes, model settings and threshold. Training and prediction times are in `metrics.json` for every model |
| Tests cover contracts, split use, weights, bit order/event counts, repeatability, and metrics | contracts: `test_wrong_column_type_is_rejected`, `test_bad_google_rows_are_rejected`; split use: `test_rows_stay_in_their_supplied_split`, `test_split_that_disagrees_with_fault_rate_is_rejected`; weights: `test_weight_counts_like_repeated_rows`; bit order and event counts: `test_toy_packing_is_little_endian_within_each_byte`, `test_unpack_detector_matrix_matches_the_single_row_helper`; repeatability: `test_task_a_is_repeatable`, `test_task_b_is_repeatable`, `test_task_c_is_repeatable`; metrics: `test_weighted_metrics_match_hand_calculation`, `test_supplied_decoder_metrics_match_hand_count` |
| Prediction IDs trace back through Part I | Every row of `predictions.parquet` keeps the `example_id` of its ML row, for all splits. Part I's `trace_examples` stage follows one syndrome and one Google `example_id` from ML to Gold, Silver and Bronze (part1.md section 9) |

The split is fixed and nothing is shuffled, the seed is fixed, and the
dependency versions are pinned (`==`) in `starter/pyproject.toml` and
`docker/requirements.txt`, so two runs on the same ML
tables give the same predictions and model files; only the timings differ.
