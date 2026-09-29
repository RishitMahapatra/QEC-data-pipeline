# Silver stage — `google_qec`

--Should we also answer the questions here?--


This section explains the main decisions taken in the Silver stage for the
Google surface-code dataset and why they were taken.

Code: [prepare_data.py](../../starter/src/quantum_lake_student/stages/prepare_data.py) ·
Tests: [test_silver_google.py](../../starter/tests/test_silver_google.py)

## Input and output

- **Input:** the Bronze archive `google-surface-code-curated.zip`, which holds
  5 experiments (four with d=3, one with d=5) of 50,000 shots each. Bronze is
  only read and never changed, and the SHA-256 of the archive is recorded.
- **Output:**
  - `silver/google_qec/experiment.parquet`: 5 rows
  - `silver/google_qec/shot.parquet`: 250,000 rows
  - Google rows in `source_trace.parquet`, `data_issues.parquet`,
    `row_counts.json` and `run.json`

## Main decisions

**1. One row per experiment folder and one row per aligned shot.**
A shot has no single source row. It is built from position `i` of 8 companion
files: 3 `b8` files and 5 `01` files. A shot is only published after all 8
files are confirmed to have exactly `shots` records, so position `i` refers to
the same shot in every file.

**2. `experiment_id` is the folder name** (for example,
`surface_code_bX_d3_r25_center_5_5`). The name is already unique, readable and
stable, so no extra key is needed.

**3. Bits are stored packed, exactly as they are in Bronze.**
`measurement_bits`, `sweep_bits` and `detector_bits` are the raw `b8` bytes of
each shot. The reasons:
- The spec asks for packed bits.
- The data is lossless and easy to verify against Bronze.
- The table stays small.

Unpacking into single bits is left to Gold/ML. When an experiment has no sweep
bits, `sweep_bits` is an empty value (`b""`) rather than null, as the spec
requires.

**4. QEC concepts are kept separate.** Measurements, detector events, the
actual logical flip and each decoder prediction are separate columns. Silver
has no "decoder mistake" column, because a mistake is derived from other
columns (`prediction != actual`) and belongs in Gold.

**5. Two levels of failure, depending on how much data the error affects.**

| Check | If it fails |
| --- | --- |
| `properties.yml` has every required key, and the values are valid integers | stop the run |
| The folder name agrees with `properties.yml` (basis, distance, rounds, center) | stop the run |
| All companion files exist | stop the run |
| Each `b8` file is exactly `shots × ceil(bits/8)` bytes long | stop the run |
| Each `01` file has exactly `shots` lines, each `0` or `1` | stop the run |
| The padding bits in a `b8` record are zero | drop that shot and log it in `data_issues.parquet` |

A wrong file length means the shots no longer line up, so no row in that
experiment can be trusted. In that case the run stops and writes nothing. A
padding error affects only one shot, so only that shot is excluded. Invalid
values are **never silently fixed**. On the supplied data every check passes
and 0 shots are rejected.

**6. Stable IDs.** `source_record_id` is a SHA-256 hash of where the record
sits in the source:
- experiment: `(source, <folder>/properties.yml)`
- shot: `(source, experiment_id, shot_index)`

Because of this, the ID does not depend on the run time or on the row order.

**7. Full tracing.** Each experiment has 1 trace row. Each shot has 8 trace
rows that share its `source_record_id`, one per companion file. The locator is
`shot_index=i` for `b8` files and `line=i+1` for `01` files. This lets you
trace any prediction back to the exact bytes and lines in Bronze.

**8. Reconciliation and reruns.**
- `row_counts.json` shows that shots read = accepted + rejected (250,000 =
  250,000 + 0).
- Silver files are overwritten on each run.
- In the shared results files, only the Google rows are replaced.

As a result, a second run gives the same records and IDs, with no duplicates.

**9. No joins with other sources.** "Distance 3" means a different code in
each dataset, so Google Silver stays on its own (see `docs/decision-log.md`).

## Requirements coverage

| Requirement | How it is met |
| --- | --- |
| Required tables, columns and types | Explicit Arrow schemas; tested |
| Validate companion-file lengths before publishing | Decision 5 |
| Stable `source_record_id` on every row | Decision 6 |
| Every row traced to Bronze | Decision 7 |
| Invalid records kept in `data_issues.parquet` | Decision 5 |
| Read = accepted + rejected; rerun gives the same output | Decision 8 |
| Concepts kept separate; no cross-source joins | Decisions 4 and 9 |

All 27 tests pass: `cd starter && uv run pytest tests/test_silver_google.py`

## Limitations

- An error that stops the run, such as a wrong file length or a missing file,
  is not recorded in `data_issues.parquet`, and one broken experiment blocks
  the other four.
- `circuit_sweep_bits` is used for the checks but is not stored as a column,
  because the spec does not require it.
