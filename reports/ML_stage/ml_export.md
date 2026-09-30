# Gold-to-ML export — `ml_syndrome_decoder_example` and `ml_google_decoder_example`

Views: [load_postgres.py](../../starter/src/quantum_lake_student/stages/load_postgres.py) (`ML_VIEWS_SQL`) ·
Export: [build_ml_tables.py](../../starter/src/quantum_lake_student/stages/build_ml_tables.py) ·
Tests: [test_build_ml_tables.py](../../starter/tests/test_build_ml_tables.py)



## 1. Design: all data logic in Gold views, a thin Python step

| Layer | Where | Does |
| --- | --- | --- |
| `v_ml_syndrome_decoder_example` | Gold view, created by `load_postgres.run()` | joins observation + experiment; `quantity` → `sample_weight`; `example_id` = the observation's primary key |
| `v_ml_google_decoder_example` | Gold view, created by `load_postgres.run()` | joins shot + experiment; pivots the four `gold_google_decoder_prediction` rows per shot back into the four `*_prediction` columns; builds `example_id` |
| `build_ml_tables.run()` | Python | `SELECT *` from each view, adds `data_split` (course helpers), checks contracts and Gold resolution, writes both Parquet files, runs the three analysis queries |

The views are created inside the same all-or-nothing transaction as the Gold
load, so they always match the loaded tables. The export reads Gold in one
`REPEATABLE READ` snapshot, so both tables and the analyses see the same data
even if a load commits meanwhile. The views end with `ORDER BY`, so on
unchanged Gold the Parquet files are byte-identical across reruns (verified on
the real data and by a test).

**`example_id` choices.**
- Syndromes: `gold_syndrome_observation.observation_id`, already a repeatable
  hash of (experiment, syndrome, label), which is the contract's definition.
  It resolves to exactly one Gold row, the table's own primary key.
- Google: `LEFT(sha256('google_qec:' || experiment_id || ':' || shot_index), 32)`,
  a repeatable (experiment, shot) hash. It resolves through the view to
  `gold_google_shot`'s primary key `(experiment_id, shot_index)`. From there,
  `source_record_id` continues to Silver and `source_trace.parquet`.

**Packed detector bytes.** `detector_bits` comes straight from Gold's `BYTEA`
column, one packed value per shot (the storage decision in
`gold_stage_google_qec.md`). A test checks it is byte-for-byte the Silver
value, so the export reproduces the packed bytes without returning to Bronze
or Silver.

## 3. Contract checks (before anything is written)

From "Checks your code must perform" in `required-ml-tables.md`:

| Check | Where |
| --- | --- |
| example IDs unique in each table | `_check_ids_and_splits` |
| train, validation, test all non-empty | `_check_ids_and_splits` |
| every syndrome has exactly 16 binary values | `syndrome_model_input` (course helper) |
| packed Google row has `ceil(detector_count / 8)` bytes, padding bits zero | `check_google_contract` |
| `detector_event_count` = number of set bits | `check_google_contract` |
| labels and decoder outputs binary, aligned by experiment and shot | non-null check + `google_meta_model_input` (course helper); the view aligns on `(experiment_id, shot_index)` |
| distance 3 and 5 kept separable (one feature width per distance) | `check_google_contract` |
| supplied model-input helpers return documented inputs | both helpers are called on every row |
| produced from Gold views, not Bronze/Silver | design above; test `test_run_never_reads_bronze_or_silver` |
| every `example_id` resolves to Gold | `gold_resolution_counts`: 0 unresolved, and ML rows = Gold rows |

Any failure raises before a Parquet file or `run.json` entry is written. Also,
pyarrow does not enforce `nullable=False` itself, so the stage checks nulls
explicitly; a missing decoder row in Gold becomes NULL in the pivot and is
rejected.

## 4. Result on the supplied release (run 2026-09-30)

| Table | Rows | Gold rows | train / validation / test |
| --- | ---: | ---: | --- |
| `ml/ml_syndrome_decoder_example.parquet` | 75,598 | 75,598 | 53,304 / 1,407 / 20,887 rows |
| `ml/ml_google_decoder_example.parquet` | 250,000 | 250,000 | 100,000 / 25,000 / 125,000 (40/10/50 %) |

The syndrome `sample_weight` sums to exactly 70,000,000. Both files read back
from MinIO with exactly the contract schema, and the supplied helpers accept
them (a distance-3 row unpacks to 200 bits). The export stage takes about 3 s.

## 5. Evidence written

- `results/part1/run.json` → `build_ml_tables`: SHA-256 of both Parquet
  files, row counts, split counts, the course data release
  (`quantum-data-core v3 (2026-08-05)`), code revision, and times.
- `results/part1/row_counts.json` → `build_ml_tables`: Gold rows vs ML rows
  and unresolved `example_id`s per table.
- `results/part1/analysis/query*_result.csv`: the three Part I analysis
  results, re-run from the committed SQL on every export, so they are
  reproducible by the pipeline instead of pasted by hand.

## 6. Limitations

- `code_revision` is the commit of `HEAD`. Inside the workspace container
  (no git binary, only `starter/` mounted) it is read from the repository's
  `.git`, mounted read-only by `compose.yaml`. Uncommitted changes cannot be
  detected that way, so a run from a modified working tree still reports the
  last commit.
- The two Parquet files are written one after the other, not atomically as a
  pair. All checks run before the first write, so only an I/O failure between
  the two writes could leave them from different runs. Their hashes in
  `run.json` would then show the mismatch.
