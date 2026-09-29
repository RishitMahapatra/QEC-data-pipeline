# Gold stage — `qec_syndromes`

This section explains the main decisions taken in the Gold stage for the
simulated surface-code syndrome dataset and why they were taken.

Code: [load_postgres.py](../../starter/src/quantum_lake_student/stages/load_postgres.py) ·
Tests: [test_load_postgres_syndromes.py](../../starter/tests/test_load_postgres_syndromes.py) ·
Analysis: [query1_weighted_syndrome_by_fault_rate.sql](query1_weighted_syndrome_by_fault_rate.sql)

## Gold model

Three tables, built from `silver/qec_syndromes/syndrome_observation.parquet`,
in schema `gold`:

| Table | One row is | Rows on the supplied data |
| --- | --- | --- |
| `syndrome_experiment` | one simulated physical-fault-rate experiment (one source CSV file) | 7 |
| `syndrome_pattern` | one distinct 16-bit syndrome pattern, seen in any experiment | 31,941 |
| `syndrome_observation` | one (experiment, syndrome pattern, logical-error label) combination and its quantity weight | 75,598 |

The same sentences are stored in PostgreSQL as comments, so they
can also be read from the database itself.

## Decision to use 3 tables

`syndrome_observation.parquet` is a single flat Silver table: every row
repeats its experiment's fault rate and its pattern's 16 bytes, and answering
something like "how many distinct patterns exist" or "which patterns are
shared across fault rates" means an ad hoc `DISTINCT` over the whole table
each time. Gold splits the same data into three entities instead: experiments,
patterns and observations as separate, queryable concepts, rather than one
copy of the Silver table:

- **`syndrome_experiment`** holds a fault-rate file's metadata
  (`physical_fault_rate`, `round_count`, `check_count`) once instead of
  repeating it across thousands of rows, so it can be queried and joined on
  its own.
- **`syndrome_pattern`** makes a syndrome pattern a real, addressable entity
  rather than sixteen repeated bytes. The same pattern is shared
  across many observations: 18,676 patterns recur under *both* labels in
  the supplied data, confirmed against the real Gold tables, so giving it
  its own row means anything computed about the pattern itself (its
  `triggered_check_count`) happens once per pattern, not once per
  observation.
- **`syndrome_observation`** ties the two together: it is the only table
  where `quantity` and `logical_error_label` live, and every row traces back
  to exactly one Silver row via `source_record_id`.

## Main decisions

**1. Normalize instead of copying Silver directly.** See "Decision to use 3
tables" above.

**2. A pattern's own bits are its key.** `syndrome_pattern.syndrome_bits` is
`BYTEA PRIMARY KEY` — there is no separate generated pattern id. The 16 bytes
are already unique per pattern (enforced by the primary key itself), so a
second identifier would only duplicate what the bytes already guarantee.

**3. `syndrome_pattern` also stores `triggered_check_count`** (how many of
the 16 bits are set), computed once per distinct pattern. This is a real,
derived attribute of the pattern itself, not just a lookup table for the
bytes, and gives the pattern table a role beyond deduplication (e.g. later
analysis of whether higher-weight patterns correlate with more logical
errors).

**4. `syndrome_observation` uses a hashed surrogate key, not its natural
composite key.** The natural key of an observation is
`(experiment_id, syndrome_bits, logical_error_label)`, enforced with a
`UNIQUE` constraint. `observation_id` is a SHA-256 hash of those same three
fields, used as the actual `PRIMARY KEY`, so that anything referencing a
specific observation later (in particular the ML export's `example_id`) only
needs to carry one column instead of three.

**5. Foreign keys connect the three tables.** `syndrome_observation`
references both `syndrome_experiment.experiment_id` and
`syndrome_pattern.syndrome_bits`. 

**6. Loading is an upsert, keyed by content, not a blind append.** Every
insert uses `ON CONFLICT`:
- `syndrome_experiment`/`syndrome_observation` use `DO UPDATE`, since their
  non-key columns (`physical_fault_rate`, `quantity`, `source_record_id`) can
  legitimately hold a newer value on a rerun and should be refreshed.
- `syndrome_pattern` uses `DO NOTHING`, since `triggered_check_count` is
  fully determined by `syndrome_bits` itself and can never need updating for
  an unchanged key.

Because `observation_id`/`experiment_id`/`syndrome_bits` are all derived
deterministically from the data rather than randomly generated, rerunning
the load on unchanged Silver data always regenerates the same keys, so the
upserts never create duplicate rows.

**7. Two failure modes an upsert alone doesn't catch, closed explicitly.**
An upsert keyed only by `observation_id` has two blind spots, both now
guarded against:
- *Two Silver rows silently colliding.* If Silver ever produced two rows
  describing the same `(experiment_id, syndrome_bits, logical_error_label)`
  but with different `source_record_id`s, both would hash to the same
  `observation_id`, and the upsert would silently keep only the last one's
  `quantity` — with no error and no record that the first one ever existed.
  `load_syndrome_tables` now checks this in Python, before touching the
  database at all: if two rows in the same batch map to the same
  `observation_id` with different `source_record_id`s, it raises `ValueError`
  and nothing is written. `syndrome_observation.source_record_id` is also
  declared `UNIQUE`, so every Gold row still traces back to exactly one
  Silver row at the schema level.
- *A row disappearing from Silver.* An upsert only ever adds or refreshes
  rows; it never removes one that used to exist. After loading the current
  batch, the load now deletes any `syndrome_observation` row (and any
  `syndrome_experiment`/`syndrome_pattern` row) whose key is not present in
  the current Silver read, so Gold stays a true mirror of the latest Silver
  output rather than a strictly-growing superset of every run there ever was.

**8. One more DB-level check beyond what Silver already validates.**
`syndrome_pattern.syndrome_bits` has `CHECK (octet_length(syndrome_bits) = 16)`
— defense in depth: even though Silver's own `pa.binary(16)` schema already
enforces this before the data ever reaches Gold, the database itself does
not just trust that nothing upstream ever changes.

**9. The load is all-or-nothing.** Schema creation, table creation and all
three load passes run inside a single transaction
(`with postgres_connection(...)`). If any step raises partway through --
including the new collision check and the new value/length `CHECK`s --
nothing from that run is committed and the previous complete Gold state is
left in place; only if every step succeeds does the transaction commit.

**10. Run evidence is recorded like every other stage.** `run()` writes a
`load_postgres`/`qec_syndromes` entry to `run.json` (the Silver file's
SHA-256, start/end time, and each Gold table's row count after load) and to
`row_counts.json` (Silver row count vs. `syndrome_observation` row count,
and `quantity` summed on both sides, overall and per experiment) via the
same `merge_json_file` helper every other stage uses.

## Requirements coverage

| Requirement | How it is met |
| --- | --- |
| State in one sentence what one row represents, for every table | Gold model table above |
| Primary keys, uniqueness rules, foreign keys, value checks, useful indexes | Decisions 2–5, 8; `CHECK`s on `physical_fault_rate`, `quantity`, `round_count`, `check_count`, `triggered_check_count`, `octet_length(syndrome_bits)`; indexes on both foreign key columns of `syndrome_observation` |
| Do not merely copy Silver into PostgreSQL | Decision 1 |
| Repeated runs leave stable identifiers, no duplicate business records, and match Silver exactly (no stale rows) | Decisions 6–7 |
| A silent data-quality error is never masked by the upsert | Decision 7 |
| All-or-nothing load | Decision 9 |
| Run/row-count evidence recorded | Decision 10 |

Tests run in a disposable, uniquely-named schema, never the real `gold`
schema, so they cannot read, write or drop another source's Gold tables; the
schema is dropped after every test, pass or fail. Tests cover: both-label
retention, pattern weight computation, idempotent reruns (no duplicates, and
quantity refreshed on change), rejection of an observation pointing at a
never-inserted experiment or pattern, rejection of a too-short pattern and a
non-positive quantity, the silent-collision check, that a failed load leaves
the previous committed data untouched, and that a row removed from Silver is
removed from Gold on the next run.

## Limitations

- The load issues one `INSERT`/`DELETE` per table per pass rather than a
  single batched/multi-row statement. The full 75,598-row load ran in
  ~14 seconds, acceptable at this scale, but would be worth batching if the
  dataset grew substantially.
- `syndrome_experiment`'s `ON CONFLICT ... DO UPDATE` only refreshes
  `physical_fault_rate`; `round_count`/`check_count` are not refreshed on
  conflict, since both are always the constant `4` on the supplied data.
- `run.json`/`row_counts.json`'s actual contents are not covered by the
  isolated-schema test suite, since verifying them means running `run()`
  itself against the real `gold` schema and real Silver data;  deliberately
  out of scope for tests that must never touch shared state. They still need
  a manual run-and-inspect check after this change, not yet done as of this
  report.
