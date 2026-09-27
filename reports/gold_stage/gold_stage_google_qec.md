# Gold Stage Documentation — `google_qec`

This document describes the Gold-stage PostgreSQL design for the `google_qec`
source: what each table represents, why the schema is shaped the way it is,
how loading is made safe to rerun, and what is explicitly out of scope.



---

## 1. Purpose

Silver produces two validated Parquet files — `experiment.parquet` and
`shot.parquet` — but Parquet is a storage format, not a queryable database.
It cannot enforce a foreign key, reject a value that violates a business
rule, or answer an ad-hoc SQL question efficiently.

The Gold stage takes Silver's already-validated output and loads it into
PostgreSQL with real relational structure: primary keys, foreign keys,
value checks, and indexes. This is the layer the three required SQL
analyses and the ML export step are built on top of.

**Pipeline position:**

<p align="center">
  <img width="732" height="446" alt="image" src="https://github.com/user-attachments/assets/ea37aaf6-12a1-4945-bf89-2604077bccad" />
</p>



---

## 2. Table inventory

Four tables make up the `google_qec` Gold model. Per table, one sentence
states what a single row represents — this mirrors the requirement that
every Gold table's grain be explicitly stated.

| Table | One row represents |
|---|---|
| `gold_google_experiment` | One Google hardware experiment configuration (a distance / rounds / processor-location combination) |
| `gold_google_shot` | One hardware shot (one circuit execution) within one experiment |
| `gold_google_decoder` | One decoder algorithm available for this source (4 rows total) |
| `gold_google_decoder_prediction` | One shot's prediction from one decoder — what it guessed, and whether it was wrong |

### Entity relationship diagram

<p align="center">
<img width="736" height="585" alt="image" src="https://github.com/user-attachments/assets/b777ba9e-4b82-4804-91a4-199dc3a609d9" />
</p>

---

## 3. Keys, foreign keys, and constraints

### Primary keys

- `gold_google_experiment.experiment_id`
- `gold_google_shot (experiment_id, shot_index)` — composite, since a shot
  index is only unique *within* its own experiment
- `gold_google_decoder.decoder_name`
- `gold_google_decoder_prediction (experiment_id, shot_index, decoder_name)`
  — composite, since one shot has one prediction per decoder

### Foreign keys

- `gold_google_shot.experiment_id → gold_google_experiment.experiment_id`
- `gold_google_decoder_prediction.decoder_name → gold_google_decoder.decoder_name`
- `gold_google_decoder_prediction (experiment_id, shot_index) → gold_google_shot (experiment_id, shot_index)`

### Generic value checks

- `distance`, `rounds`, `shots`, `measurement_count`, `detector_count` must
  all be positive
- `shot_index` and `detector_event_count` must be non-negative

### Data-driven checks — rules that come from the data itself, not just "> 0"

These four were added specifically because a generic positivity check
would let internally-contradictory rows through:

1. **Detector count must match code distance.** A distance-3 experiment
   always has exactly 200 detectors; distance-5 always has exactly 600.
   Enforced as a table-level `CHECK` combining both columns — a row
   claiming `distance = 3` and `detector_count = 600` is rejected outright.
2. **Basis restricted to X or Z.** A surface-code memory experiment is
   only ever measured in one of these two bases; anything else is a data
   error, not a valid third option.
3. **No duplicate processor-location configuration.** A `UNIQUE` constraint
   on `(distance, rounds, center_row, center_col)` prevents two experiment
   rows from silently claiming the same physical chip location and
   settings — which would make them indistinguishable.
4. **Packed detector-byte length matches distance.** `detector_bits` must
   be exactly 25 bytes (distance-3) or 75 bytes (distance-5) — a defensive
   check independent of Silver, catching any bug introduced between the
   two stages rather than blindly trusting upstream validation.

### Indexes

- `idx_google_shot_experiment_id` — supports every join between shot and
  experiment
- `idx_decoder_prediction_decoder` — supports grouping/filtering by decoder
  (e.g. "error rate per decoder")

---

## 4. Detector-bits storage decision

`detector_bits` is stored as **one packed `BYTEA` value per shot row**, not
as one database row per individual detector reading.

**The numbers:**

- Distance-3: 200 detectors/shot → `ceil(200/8) = 25` bytes packed
- Distance-5: 600 detectors/shot → `ceil(600/8) = 75` bytes packed
- Across all 250,000 shots (4 distance-3 experiments × 50,000 shots + 1
  distance-5 experiment × 50,000 shots):
  - `(200,000 × 25) + (50,000 × 75) = 5,000,000 + 3,750,000 = 8,750,000`
    bytes ≈ **8.75 MB total**
- The alternative — one row per individual detector reading — would
  produce `(200 × 200,000) + (600 × 50,000) = 40,000,000 + 30,000,000 =`
  **70,000,000 rows**, a **~280×** increase in row count.

No required SQL analysis or the ML export table needs single-detector
granularity — both only ever consume the packed bits and a precomputed
`detector_event_count`. The compact design was chosen because it matches
what is actually used, and avoids an unjustified storage/complexity cost.

<p align = "center">
<img width="696" height="623" alt="image" src="https://github.com/user-attachments/assets/e247ea98-11c8-4794-be46-eebdd5eba8e1" />
</p>

---

## 5. Why this is not a copy of Silver: the decoder redesign

Silver's `shot.parquet` correctly stores each decoder's prediction as its
own flat boolean column — that is the right call for Silver, where the
goal is simply validated, one-row-per-shot data.

Gold's purpose is different: it is meant to be a relational model built
for querying. Carrying the same four flat columns forward would mean Gold
is just a copy of Silver with keys bolted on — which the rubric explicitly
warns against.

**The redesign:** the four decoder prediction columns were split out of
`gold_google_shot` into two new tables:

- `gold_google_decoder` — one row per decoder name (4 rows)
- `gold_google_decoder_prediction` — one row per shot per decoder, with
  `predicted_flip` (what it guessed) and `was_wrong` (whether that
  disagreed with the actual outcome) as two separate, explicit columns

**Why this is better:**

- "What a decoder guessed" and "whether it was wrong" are now distinct
  facts instead of something a reader has to recompute by comparing two
  columns themselves every time.
- Adding a fifth decoder in the future needs no schema change — just more
  rows, not a new column.
- "Error rate per decoder" becomes a single grouped query instead of four
  separately written, copy-pasted queries.

<p align = "center">
<img width="766" height="354" alt="image" src="https://github.com/user-attachments/assets/2078fbd0-8f60-4557-87cf-8246baa0d0a7" />
</p>

---

## 6. Load behavior: all-or-nothing and rerun-safety

Every load runs inside a single transaction:

1. Ensure the schema exists (`CREATE TABLE IF NOT EXISTS` — safe to repeat)
2. `TRUNCATE` all four tables (`RESTART IDENTITY CASCADE`, so dependency
   order is handled automatically and no stale rows survive)
3. Bulk-insert experiment rows, then shot rows, then decoder rows, then
   decoder-prediction rows
4. Commit only if every step above succeeded

If any step raises an exception, the caller rolls back the entire
transaction — including the earlier `TRUNCATE` — so a failed load leaves
the **previous complete Gold data untouched**, never a half-loaded state.

Because every load truncates before inserting, running the load twice in
a row produces identical row counts, not duplicates.

<p align = "center">
<img width="933" height="530" alt="image" src="https://github.com/user-attachments/assets/898d0fe9-d801-43db-804b-175f611d9edb" />
</p>

---

## 7. Row-count reconciliation

Rather than reporting a single "rows loaded" number, `row_counts.json`
records what Silver had and what Gold actually loaded side by side, with
a computed match flag — so a silent drop between the two stages would be
visible, not hidden:

```json
{
  "experiment": { "silver_rows": 5, "gold_loaded": 5, "match": true },
  "shot": { "silver_rows": 250000, "gold_loaded": 250000, "match": true },
  "decoder_prediction": { "expected": 1000000, "gold_loaded": 1000000, "match": true }
}
```

---

## 8. Test coverage

The Gold stage is covered by 14 tests, run against a real, isolated
throwaway Postgres schema (created and dropped per test — the real
`gold_google_*` tables are never touched):

- Schema creation (all four tables exist)
- Both foreign key relationships reject orphaned rows
- All four data-driven constraints, both the reject case and the valid
  case
- `source_record_id` round-trips correctly from a Silver row to its Gold
  row (traceability)
- Decoder-prediction `was_wrong` is computed correctly against the actual
  outcome
- Repeatability: loading twice produces identical row counts, not
  duplicates
- **All-or-nothing:** a successful load followed by a deliberately broken
  second load correctly rolls back, leaving the original data completely
  intact

---
## Requirements coverage

| Requirement | How it is met |
| --- | --- |
| Student-designed tables, not a copy of Silver | Decoder table split (Section 5) |
| Primary keys, foreign keys, value checks, indexes | Section 3 |
| Rules driven by the data, not generic checks | Section 3, "Data-driven checks" |
| All-or-nothing load | Section 6 |
| Repeated loads create no duplicates | Section 6 (truncate-then-insert) |
| Every row traceable to its Silver source | `source_record_id` on experiment and shot (Section 2) |
| Reconciled row counts, not just a single number | Section 7 |
| No invented cross-source relationship | Section 9 |

All 14 tests pass: `cd starter && uv run pytest tests/test_load_postgres.py`

## 9. Explicitly rejected relationship

Gold for `google_qec` does **not** join to `qasmbench` or `qec_syndromes`
at the row level. No source-record-level identifier is shared between
these sources — a QASMBench circuit statement and a Google hardware shot
describe fundamentally different things with no common key. This is an
intentional, evidence-based design decision, not an oversight.

---

## 10. Open items (team-level, not resolved in this document)

- Whether Gold tables should live in a dedicated `gold` schema instead of
  `public`
- Shared cross-source vocabulary (e.g. a common `code_family` table, a
  consistently-named `distance` column across all three sources) —
  needed before any cross-source query or comparison can be written
