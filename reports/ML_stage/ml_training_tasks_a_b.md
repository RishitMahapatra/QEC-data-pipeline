# ML training stage — Tasks A and B

This section explains how Part II Tasks A and B are built, which decisions
were taken and why, and what the results show.

Code: [train.py](../../starter/src/quantum_lake_student/stages/train.py) ·
Tests: [test_train.py](../../starter/tests/test_train.py)

## Input and output

- **Input:** only the two ML tables produced by Part I (see
  [ml_export.md](ml_export.md)):
  - `ml/ml_syndrome_decoder_example.parquet`: 75,598 rows, 70,000,000
    weighted shots (Task A)
  - `ml/ml_google_decoder_example.parquet`: 250,000 hardware shots (Task B)

  Gold, Silver and Bronze are never read, and nothing is written back to the
  lake.
- **Output:** everything under `results/part2/`, rebuilt by one command
  (`make train`):
  - `models/`: the 6 fitted models (`.pkl`)
  - `predictions.parquet`: 1,651,196 rows, one per example per model
  - `metrics.json`: test metrics per task (and per distance for Task B)
  - `run.json`: run id, code revision, data release, SHA-256 of both input
    tables, seed, library versions, feature order, split rule, thresholds,
    learned coefficients and timings
  - `report.md`: the short Part II report, generated from the same numbers

## How it runs

```
make train
  -> read both ML tables, hash them
  -> Task A: validate -> build features -> fit on train -> threshold on validation -> score on test
  -> Task B: validate -> for d=3 and d=5 separately: same steps + decoder overlap
  -> write models, predictions, metrics, run.json, report.md
```

## Main decisions (shared)

**1. Part II checks its own input.** Each table is validated against
`required-ml-tables.md` before any training: exact columns and types, no
nulls, unique `example_id`, all three splits present, and the split of every
row matching the course helper. Part I already checks this when it exports,
but the consumer does not just trust the producer. If anything fails, the run
stops and nothing is written.

**2. Inputs only come from the course helpers.** Task A uses
`syndrome_model_input` (16 bits) and Task B uses `google_meta_model_input`
(event density + 4 predictions). No extra features are built.

**3. Train fits, validation chooses, test scores.** Models are fitted on
train only. The one choice we make, the decision threshold, is picked on
validation from 0.01 to 0.99 (lowest error rate, ties go to the value closest
to 0.5). Test is only used once, for the reported metrics. Tests prove that
flipping every test label does not change the fitted model or the threshold.

**4. Every model gets a prior baseline.** A `DummyClassifier(strategy="prior")`
always predicts the most common label and gives the train share as its
probability. It is the bar the other models are compared against.

**5. Repeatable by construction.** The splits are fixed, logistic regression
(`lbfgs`) is deterministic, the seed is fixed (`4000`) and nothing is
shuffled. Two runs give byte-identical predictions and model files; only the
timings change.

**6. Predictions are kept for every split, with `example_id`.** Metrics only
use test, but saving all splits means any example in the ML tables can be
traced from its prediction back to Gold, Silver and Bronze through Part I's
`trace_examples` stage.

## Task A — weighted syndrome decoder

**Goal.** Predict `logical_error_label` from the 16 syndrome bits (4 rounds ×
4 checks, ordered by round then check).

**7. Weights are used everywhere.** One row stands for up to ~10M identical
shots, so both models are fitted with `sample_weight`, and every metric
(error rate, balanced accuracy, Brier score) is weighted the same way. The
rows are never expanded into 70M observations. A test checks that a row with
weight 2 fits exactly like two rows with weight 1.

**8. One linear model: logistic regression.** The task allows exactly one
weighted linear classifier. Logistic regression supports `sample_weight`,
gives a probability (so a Brier score exists), and its 16 coefficients are
easy to read.

**Splits** (fixed by fault rate, used unchanged):

| Split | Fault rates | Rows | Shots | Error share |
| --- | --- | ---: | ---: | ---: |
| train | 1e-5, 5e-5, 1e-4, 0.001, 0.01 | 53,304 | 50M | 4.3% |
| validation | 0.0005 | 1,407 | 10M | 1.1% |
| test | 0.005 | 20,887 | 10M | 10.4% |

**Results on test** (all weighted):

| Model | Logical-error rate | Balanced accuracy | Brier score |
| --- | ---: | ---: | ---: |
| Weighted prior | 0.1040 | 0.5000 | 0.0969 |
| Weighted logistic regression (threshold 0.93) | 0.1041 | 0.5008 | 0.0910 |

**Findings.**
- The logistic regression does not beat the prior on error rate. Its
  probabilities are better (lower Brier), so it did learn something, but its
  yes/no answers are almost the same as "always no error".
- The reason is the threshold. Validation has only 1.1% errors, so the
  threshold with the lowest validation error is 0.93, which almost never says
  "error". Test has about 9× more errors, so that choice does not carry over.
  We report this as it is instead of tuning on test.
- The coefficients make sense: every check except two has a positive weight,
  and the checks in column 0 and 2 matter most (e.g. `round0_check2` +2.29,
  `round0_check0` +1.73, intercept −4.03).
- A linear model adds each bit's effect on its own, so it cannot see the bit
  combinations that signal a logical error. This is the main limit of Task A.

## Task B — supplied and combined Google decoders

**Goal.** On real hardware shots, compare the four supplied decoders and one
combined decoder, separately for distance 3 and distance 5.

**9. Distances are never mixed.** Distance 3 (200 detectors, 4 experiments,
200,000 shots) and distance 5 (600 detectors, 1 experiment, 50,000 shots) are
filtered apart before anything else, and each gets its own models,
threshold, metrics and report section.

**10. Extra checks on the Google table.** On top of decision 1, every packed
row must have `ceil(detector_count / 8)` bytes with zero padding bits,
`detector_event_count` must equal the number of fired bits, the detector
count must be 200 for d=3 and 600 for d=5, and `(experiment_id, shot_index)`
must be unique.

The supplied data has no padding bits at all: 200 and 600 are both multiples
of 8, so every packed row fills its 25 or 75 bytes exactly and the padding
check can never fail on it. The check is still there because the contract
names it, and it is tested with a made-up count of 12 detectors (2 bytes, 4
padding bits).

**11. Supplied decoders are scored, not run.** Their predictions are read
from the same row as the label, so all models are compared on identical test
shots (a test checks this). They only give yes/no answers, so their Brier
score and timings are recorded as not applicable rather than invented.

**12. One combined decoder: logistic regression on 5 inputs.** It learns how
much to trust each decoder and how the shot's noise level (event density)
shifts the answer. The threshold is chosen on validation (0.51 for d=3, 0.55
for d=5).

**13. Decoder overlap is measured.** For every test shot we count how many
decoders are wrong, how often each pair is wrong together, and which shots
the combined model fixes or breaks compared with the best single decoder.

**Results on test** (logical-error rate / balanced accuracy):

| Model | Distance 3 | Distance 5 |
| --- | ---: | ---: |
| Prior | 0.4929 / 0.500 | 0.5015 / 0.500 |
| PyMatching | 0.4364 / 0.564 | 0.4548 / 0.545 |
| Correlated matching | 0.4220 / 0.578 | 0.4226 / 0.577 |
| Belief matching | 0.4013 / 0.599 | 0.3994 / 0.601 |
| Tensor network contraction | **0.3967** / 0.603 | **0.3952** / 0.605 |
| Combined logistic regression | 0.3977 / 0.602 | 0.3977 / 0.603 |

The prior has a Brier score of 0.250 and the combined model 0.235 (d=3) and
0.234 (d=5).

**Decoder overlap on test:**

| Decoders wrong on a shot | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distance 3 | 30.2% | 19.0% | 20.2% | 16.1% | 14.4% |
| Distance 5 | 24.4% | 23.7% | 23.1% | 18.0% | 10.9% |

**Findings.**
- All four decoders beat the prior, but they are still wrong on about 40% of
  shots. Each shot covers 25 rounds, so the final logical outcome is close
  to 50/50.
- In most pairs, the decoders make different mistakes more often than the
  same ones. They disagree on 55% of shots (d=3) and 65% (d=5). At d=5 every
  pair shares less than half of its mistakes (33% to 49%). At d=3 that holds
  for four of the six pairs (36% to 48%); the other two, belief matching with
  tensor network and correlated matching with PyMatching, share slightly more
  than half (53%). Tensor network and PyMatching overlap the least.
- The combined model ties the best decoder (tensor network) but does not beat
  it. At d=3 it fixes 6.6% of the shots tensor network gets wrong and breaks
  6.7% it got right; at d=5 it fixes 5.3% and breaks 5.6%. The 14.4% / 10.9%
  of shots where all four are wrong cannot be fixed by any combination of
  their answers.
- The learned coefficients match the ranking: tensor network gets the
  largest weight (+0.53 at d=3, +0.59 at d=5) and PyMatching the smallest
  (+0.17, +0.22). Event density matters more at d=5 (+1.10) than at d=3
  (+0.38).
- Distance 5 is not clearly better than distance 3 here, and since it comes
  from one experiment at one chip location, we do not draw a conclusion
  about code size from it.

## Requirements coverage

| Requirement (part-2-ai-ml.md) | How it is met |
| --- | --- |
| Read only the two ML tables | Input section; a test checks only those two files are read |
| Validate both input schemas before training | Decisions 1, 10 |
| Use only features from the supplied helpers | Decision 2 |
| Train for fitting, validation for choices, test once | Decision 3 |
| Task A: weighted prior + one weighted linear classifier | Decisions 4, 7, 8 |
| Physical weights in fitting and all Task A metrics | Decision 7 |
| Task B: d=3 and d=5 separate | Decision 9 |
| Task B: four decoders on the same test shots, Brier n/a | Decision 11 |
| Task B: linear combined decoder on the 5 inputs | Decision 12 |
| Task B: explain same or different mistakes | Decision 13, findings |
| Error rate, balanced accuracy, Brier, training and prediction time | `metrics.json`, `report.md` |
| Fixed seed and versions, one command, saved models | Decision 5, `run.json`, `models/` |
| `predictions.parquet` with `example_id` | Decision 6 |
| Tests for columns/types, splits, weights, bit order, event counts, padding, repeatability, metrics | 38 tests for Tasks A and B in `test_train.py` |

All tests pass:
`docker compose exec workspace pytest tests/test_train.py -v`

## Limitations

- **`uv.lock` is behind the pins.** `numpy==2.5.3` and `scipy==1.18.1` are
  now pinned in `docker/requirements.txt` and `pyproject.toml`, matching the
  container. `uv.lock` still lists numpy 2.5.1 and scipy 1.18.0 and has to be
  regenerated with `uv lock`. The container installs from `requirements.txt`,
  so results are not affected.
- **`run.json` records the last commit, not uncommitted changes.** The code
  has to be committed before the final run so the revision matches.
- **Tracing order.** `make run` (Part I, including `trace_examples`) runs
  before `make train`, so on a clean run the trace does not show a prediction
  yet. `make run` has to be repeated after training, or the trace stage run
  again, to include it.
- **Task A threshold.** Picking the threshold by lowest validation error
  makes the model act almost like the prior on test. It is a correct result,
  but a balanced-accuracy threshold would make it a more useful detector.
- **Linear models are the main limit.** Neither task's linear model can use
  bit or vote combinations; Task C (a small MLP on raw detector bits) is the
  first non-linear model and is not covered here.
