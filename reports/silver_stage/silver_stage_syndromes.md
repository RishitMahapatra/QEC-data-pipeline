# Silver stage — `qec_syndromes`

This section explains the main decisions taken in the Silver stage for the
simulated surface-code syndrome dataset and why they were taken.

Code: [prepare_data.py](../../starter/src/quantum_lake_student/stages/prepare_data.py) ·
Tests: [test_silver_syndromes.py](../../starter/tests/test_silver_syndromes.py)

## Input and output

- **Input:** the Bronze archive `syndromes_dataset.zip`, which holds 7 CSV
  files, one per distance-3 physical fault rate, each an aggregate of
  ~10,000,000 simulated shots. Bronze is only read and never changed, and the
  SHA-256 of the archive is recorded.
- **Output:**
  - `silver/qec_syndromes/syndrome_observation.parquet`: 75,598 rows
  - qec_syndromes rows in `source_trace.parquet`, `data_issues.parquet`,
    `row_counts.json` and `run.json`

## Main decisions

**1. One row per original aggregate CSV row.**
Unlike the Google dataset, a syndrome observation is fully self-contained in
a single CSV line — there are no companion files to align, so each accepted
CSV row maps to exactly one Silver row.

**2. `experiment_id` is the source filename** (for example,
`qec_syndromes:d-3_pfr-0.001000_nb-10M.csv`). The filename already encodes the
physical fault rate and is stable and unique per file, so no extra key is
needed. `physical_fault_rate` itself is parsed out of the same filename.

**3. `syndrome_bits` is stored as 16 one-byte values, round-major then
check-second** (4 rounds × 4 checks). The `syndromes` column arrives as
Python-literal nested-tuple text, e.g.
`"((0, 0, 0, 0), (0, 0, 1, 0), ...)"`. It is parsed with `ast.literal_eval`
(safe against arbitrary code, unlike `eval`), never with plain string
splitting, and rejected as invalid if it is not exactly 4 tuples of 4 binary
values.

**4. `quantity` is kept as a weight, never expanded into repeated rows.** A
single row can represent millions of physical observations; multiplying it
out into one row per observation would make Silver ~70M rows instead of
~76k, and would throw away the fact that some observations are far more
common than others.

**5. The same syndrome pattern is never deduplicated across labels.** A
pattern that legitimately occurs with both `logical_error_label = 0` and
`= 1` produces two separate accepted rows, one per label. This is verified
directly with a fixture built around exactly this collision.

**6. Two tiers of failure, at the file level and the row level.**

| Check | If it fails |
| --- | --- |
| Filename matches `d-<distance>_pfr-<rate>_nb-<n>.csv` | file skipped, logged |
| Header line is exactly `labels,syndromes,quantity` | file skipped, logged |
| File has at least a header row | file skipped, logged |
| Row splits into exactly `label,syndromes,quantity` | row excluded, logged |
| `syndromes` parses into a 4×4 binary tuple | row excluded, logged |
| `quantity` is a positive integer | row excluded, logged |
| `labels` is `0` or `1` | row excluded, logged |
| Accepted `quantity` total for a file is within 2% of the `nb-<n>` filename hint | file **flagged as a warning**; already-accepted rows are kept |

Unlike a run-stopping error, every one of these is recorded in
`data_issues.parquet` with a `rule_id`, `severity`, `observed_value` and
`reason` — including file-level failures, which use a `null`
`source_record_id` since no single row caused them. A malformed file does not
stop the other 6 files from being processed. On the supplied data every
check passes and 0 rows are rejected.

**7. Stable IDs.** `source_record_id` is a SHA-256 hash of
`(source filename, row position)`, truncated to 32 hex characters. Because it
depends only on where the row sits in its source file, not on run time or
row order, the same row always gets the same ID on every rerun.

**8. Full tracing, including rejected rows.** Every row that is read gets a
trace row the moment it is split out of the CSV, before any validation runs,
so even a row that later gets excluded still has a `source_trace.parquet`
entry pointing at its exact file and line number. A rejected row's trace and
its `data_issues.parquet` entry can therefore both be found by the same
`source_record_id`.

**9. Reconciliation and reruns.**
- `row_counts.json` records `accepted`/`rejected` per run.
- `run.json` records the Bronze input hash and per-table output counts.
- Silver output is overwritten on each run; in the shared results files
  (`source_trace.parquet`, `data_issues.parquet`), only qec_syndromes' own
  rows are replaced, leaving other sources' rows untouched.

As a result, a second run gives the same records and IDs, with no
duplicates.

**10. No joins with other sources.** qec_syndromes rows are parsed and
validated entirely on their own; nothing here depends on google_qec or
qasmbench, and no cross-source identifier is invented to connect them.

## Requirements coverage

| Requirement | How it is met |
| --- | --- |
| Required columns and types | Explicit Arrow schema (`SYNDROMES_SILVER_SCHEMA`); tested |
| 4×4 shape and binary-bit validation | Decisions 3, 6 |
| `quantity` kept as a weight, not expanded | Decision 4 |
| Same syndrome under both labels is retained, not deduped | Decision 5 |
| Weighted totals reconcile to the filename hint | Decision 6 |
| Invalid records kept in `data_issues.parquet` | Decision 6 |
| Stable `source_record_id` on every row | Decision 7 |
| Every row traced to Bronze, valid or not | Decision 8 |
| Read = accepted + rejected; rerun gives the same output | Decision 9 |
| No cross-source joins | Decision 10 |

All 5 tests pass: `docker compose exec workspace pytest tests/test_silver_syndromes.py -v`

**Result on the supplied data:** all 7 files are read and accepted, 75,598
rows are accepted, and 0 rows/files are rejected. The total accepted
`quantity` across all 7 files is ~70,000,000, matching the ~10M-per-file
weighted-shot count from the course release.

## Limitations

- The weighted-total reconciliation check is a warning only — a file whose
  accepted `quantity` total drifts from its filename hint by more than 2% is
  flagged in `data_issues.parquet`, but its already-valid rows are still
  loaded rather than the whole file being rejected.
- `round_count`/`check_count` are always written as the constant `4` rather
  than derived from the parsed structure's actual length, since a row can
  only reach that point in the code after already being validated as exactly
  4×4.
