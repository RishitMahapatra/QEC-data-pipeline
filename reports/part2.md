# PART 2 REPORT


## PART A



## PART B



## PART C

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
- "You are not required to design or research a more complex model" — the
  task is explicitly a **pipeline-consumer check**: can the prepared data
  feed a model that expects raw per-detector inputs, not "build the best
  decoder."

## How we decided to implement it

**Why a fixed small MLP, not a search.** The spec says explicitly this
isn't an architecture competition and model performance isn't graded. So
we picked one small, fixed architecture (`hidden_layer_sizes=(32,)`,
`max_iter=300`) up front and did not tune it — tuning would spend effort
the rubric doesn't reward and risks implicitly turning this into a
competition the spec says to avoid.

**Why we built our own bit-unpacking helper.** Tasks A and B use
course-supplied helpers (`syndrome_model_input`, `google_meta_model_input`)
to turn packed bytes into model inputs. No course helper does this for
*raw* detector bits at the per-bit level Task C needs, so we wrote
`unpack_detector_matrix()` ourselves, using `numpy.unpackbits(...,
bitorder="little")` — the same little-endian-per-byte convention Part I
used to pack the bits in the first place (documented in the Gold design
doc, Section 6.5). We reused the existing convention rather than inventing
a new one, so Task C's inputs stay consistent with how the rest of the
pipeline reads these bytes.

**Why we filter before unpacking, not after.** `raw_detector_splits()`
filters the table to `distance == 3 AND shot_index < 12_500` first, then
groups by the *existing* `data_split` column. We don't recompute splits
ourselves — the split was already assigned by Part I (`google_data_split`)
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

## Rubric comparison (item 8 — Raw-detector prototype, 6 points)

| Rubric requirement (5–6 point band) | How Task C meets it |
| --- | --- |
| The required bounded distance-three MLP runs on the fixed subset | `RAW_DETECTOR_DISTANCE = 3`, `RAW_DETECTOR_SHOT_LIMIT = 12_500`; `raw_detector_splits()` filters to exactly `distance == 3 AND shot_index < 12_500` before anything else happens |
| Bit order and split are correct | `unpack_detector_matrix()` uses `bitorder="little"`, matching Part I's packing convention; split comes from the already-validated `data_split` column, not recomputed |
| Training/prediction time is recorded | `run_task_c()` times `.fit()` once and `.predict_proba()` per split, stored in `metrics["models"][MLP_MODEL_ID]` |
| The report clearly explains information hidden by a flat input | `task_c_report()`'s closing paragraph names spatial adjacency and temporal (across-round) structure as what gets discarded, and explains why that caps a flat-vector model regardless of tuning |

Nothing here claims or needs a specific accuracy number — rubric item 8,
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
