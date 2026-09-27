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
| `syndrome_experiment` | one physical-fault-rate simulation file | 7 |
| `syndrome_pattern` | one distinct 16-bit syndrome pattern, seen anywhere in the dataset | (distinct across all 7 files) |
| `syndrome_observation` | one (experiment, pattern, label) combination and its weight | 75,598 |

## Main decisions

**1. Normalize instead of copying Silver directly.** `syndrome_observation.parquet`
is one flat table; Gold splits it into experiment, pattern and observation
instead of loading it as-is, so that "which patterns are shared across fault
rates" or "how many distinct patterns exist" are real joins/aggregates
against a modeled entity, not incidental byproducts of a flat table.

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

**7. The load is all-or-nothing.** Schema creation, table creation and all
three load passes run inside a single transaction
(`with postgres_connection(...)`). If any step raises partway through,
nothing from that run is committed and the previous complete Gold state is
left in place; only if every step succeeds does the transaction commit.

## Requirements coverage

| Requirement | How it is met |
| --- | --- |
| State in one sentence what one row represents, for every table | Gold model table above |
| Primary keys, uniqueness rules, foreign keys, value checks, useful indexes | Decisions 2–5; `CHECK`s on `physical_fault_rate`, `quantity`, `round_count`, `check_count`, `triggered_check_count`; indexes on both foreign key columns of `syndrome_observation` |
| Do not merely copy Silver into PostgreSQL | Decision 1 |
| Repeated runs leave stable identifiers, no duplicate business records | Decision 6 |
| All-or-nothing load | Decision 7 |

Tests run in a disposable, uniquely-named schema, never the real `gold`
schema, so they cannot read, write or drop another source's Gold tables; the
schema is dropped after every test, pass or fail.

## Limitations

- The load issues one `INSERT` per row (three passes: experiments, patterns,
  observations) rather than a single batched/multi-row statement. This ran
  in ~14 seconds against the full 75,598-row dataset, which is acceptable at
  this scale, but would be worth batching if the dataset grew substantially.
- `syndrome_experiment`'s `ON CONFLICT ... DO UPDATE` only refreshes
  `physical_fault_rate`; `round_count`/`check_count` are not refreshed on
  conflict, since both are always the constant `4` on the supplied data.
