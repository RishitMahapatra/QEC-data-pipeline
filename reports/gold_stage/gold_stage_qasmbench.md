# Gold Stage Documentation — `qasmbench`

This document describes the Gold-stage PostgreSQL design for the `qasmbench`
source: what each table represents, why the schema is shaped the way it is,
how loading is made safe to rerun, and what is explicitly out of scope. It
follows the same structure as
[`gold_stage_google_qec.md`](gold_stage_google_qec.md), and the code lives in
the same `load_postgres.py` module, in the same shape (`load_<source>_gold`
+ `run_<source>_gold`) — this document only covers what is different about
qasmbench's requirements and design, not a re-explanation of the shared
pattern.

---

## 1. Purpose

Silver produces three validated Parquet files — `circuit.parquet`,
`stabilizer_check.parquet`, and `conditional_correction.parquet` — but
Parquet cannot enforce a foreign key, reject a value that violates a
business rule, or answer an ad-hoc SQL question efficiently.

The Gold stage takes Silver's already-validated output and loads it into
PostgreSQL with real relational structure: primary keys, foreign keys, value
checks, and indexes. This is the layer the "how does the repetition-code
circuit map data qubits to parity-check ancillas, syndrome bits, and
conditional corrections?" analysis question (`assignment/brief.md`, Part I
analysis question 3) is built on top of.

Code: [`load_postgres.py`](../../starter/src/quantum_lake_student/stages/load_postgres.py) ·
Tests: [`test_gold_qasmbench.py`](../../starter/tests/test_gold_qasmbench.py),
[`test_load_postgres_run.py`](../../starter/tests/test_load_postgres_run.py)

---

## 2. Table inventory

Four tables make up the `qasmbench` Gold model — one more than Silver's
three, because `register_declarations` is decomposed rather than copied (see
Section 5).

| Table | One row represents |
|---|---|
| `gold_qasmbench_circuit` | One parsed QASMBench circuit variant (a benchmark, as published or after transpilation) |
| `gold_qasmbench_register` | One declared quantum or classical register of one circuit variant |
| `gold_qasmbench_stabilizer_check` | One explicit parity-check relationship (ancilla, its data qubits, and the resulting syndrome bit) found in one circuit variant |
| `gold_qasmbench_conditional_correction` | One syndrome-conditioned correction gate applied to one target qubit in one circuit variant |

---

## 3. Keys, foreign keys, and constraints

### Primary keys

- `gold_qasmbench_circuit.circuit_id` — already unique and stable
  (`"<benchmark>:<variant>"`)
- `gold_qasmbench_register (circuit_id, register_kind, register_name)` —
  composite, since a register name is only unique within its own kind,
  within its own circuit
- `gold_qasmbench_stabilizer_check.source_record_id`
- `gold_qasmbench_conditional_correction.source_record_id`

### Foreign keys

- `gold_qasmbench_register.circuit_id → gold_qasmbench_circuit.circuit_id`
- `gold_qasmbench_stabilizer_check.circuit_id → gold_qasmbench_circuit.circuit_id`
- `gold_qasmbench_conditional_correction.circuit_id → gold_qasmbench_circuit.circuit_id`

All three are declared `ON DELETE CASCADE`, so the truncate-and-reload in
Section 6 clears every dependent table in one statement.

### Generic value checks

- `qubit_count` must be positive; `measurement_count` and
  `two_qubit_gate_count` must be non-negative
- `register_size` must be positive
- `condition_value` must be non-negative

### Data-driven checks — rules that come from the data itself, not just "> 0"

Google_qec's equivalent checks are about cross-column facts (distance ↔
detector count, packed-byte length ↔ distance) — qasmbench's archive has no
such cross-column relationship to enforce. Its data-driven rules are instead
about *shape*:

1. **`variant` is restricted to `'source'` or `'transpiled'`.** Anything else
   is a data error, not a valid third option — mirrors google_qec's
   `basis IN ('X','Z')` check.
2. **`register_kind` is restricted to `'qreg'` or `'creg'`.** A QASM register
   is one or the other; nothing else is meaningful here.
3. **`data_qubits` must never be empty.**
   `CHECK (cardinality(data_qubits) >= 1)` — a stabilizer check with zero
   data qubits is not a parity check. Note this uses `cardinality()`, not
   `array_length(data_qubits, 1)`: `array_length` of an empty array returns
   `NULL`, and Postgres treats a `NULL` `CHECK` result as passing (not
   failing), so an `array_length`-based version of this constraint silently
   let an empty array through — caught by
   `test_stabilizer_check_rejects_empty_data_qubits` during testing (see
   Section 8) and fixed by switching to `cardinality()`, which returns `0`
   for an empty array.
4. **No duplicate `(benchmark_name, variant)`, independently of `circuit_id`.**
   A `UNIQUE` constraint prevents two circuit rows from silently claiming
   the same benchmark+variant combination — mirrors google_qec's
   `UNIQUE(distance, rounds, center_row, center_col)`.

### Indexes

- `idx_qasmbench_register_circuit_id`
- `idx_qasmbench_stabilizer_check_circuit_id`
- `idx_qasmbench_conditional_correction_circuit_id`

Postgres does not index foreign-key columns automatically; all three support
joining back to `gold_qasmbench_circuit`, which the "map data qubits to
ancillas, syndrome bits, and corrections" analysis query needs.

---

## 4. Why table names and file location match google_qec

Both sources' Gold tables now live in the same `load_postgres.py`, merged
from two independent lines of work (qasmbench's own implementation, and
google_qec's from the team's `rishit_mahapatra` branch). Rather than leave
the two sources visibly inconsistent — one schema-qualified
(`qec.qasmbench_*`), one not (`gold_google_*`) — qasmbench's tables were
renamed to `gold_qasmbench_*` and moved out of a dedicated `qec` schema to
match google_qec's existing, already-tested convention. See
`docs/decision-log.md`'s "qasmbench Gold design" section for the full
reasoning; which schema the whole Gold model should eventually live in is
still an open, team-level decision (`gold_stage_google_qec.md` Section 10).

---

## 5. Why this is not a copy of Silver: the register decomposition

Silver's `circuit.parquet` correctly keeps `register_declarations` as one
JSON-text column (a nested list of qreg/creg name+size pairs) — that is the
right call for Silver, where the goal is simply validated, one-row-per-circuit
data.

Gold's purpose is different: it is meant to be a relational model built for
querying and constraining. Carrying the same opaque JSON blob forward, or
even storing it as a queryable `jsonb` column *alongside* a normalized copy,
would mean Gold is either a copy of Silver with keys bolted on, or two
representations of the same fact with nothing keeping them in sync — the
rubric explicitly warns against the former, and the latter is worse.

**The redesign:** `register_declarations` is decomposed into its own table,
`gold_qasmbench_register` — one row per declared register, with
`register_kind`, `register_name`, and `register_size` as typed, constrained
columns instead of buried JSON fields.

**Why this is better:**

- A register's size is now a queryable, constrained (`register_size > 0`)
  fact, not something a reader has to parse out of a JSON string every time.
- Answering "how many qubits does the ancilla register have" is a direct
  `SELECT`, not JSON-path extraction.
- It costs one small table (4 rows per circuit variant: `q`, `a`, `c`, `syn`
  for the parity-check benchmark) — decomposing does not add real
  complexity for data this size.

`data_qubits` (on `gold_qasmbench_stabilizer_check`) was *not* given the
same treatment: it stays a native `text[]` column rather than a fifth
junction table, because — unlike `register_declarations`, which described
multiple independent registers — `data_qubits` is a small (2 elements),
fixed-shape, ordered attribute of a single check row (~8 total (qubit,
check) pairs across the whole archive). A junction table there would add a
whole table for a handful of rows with no independent identity of their
own. See `docs/decision-log.md` for the full trade-off.

---

## 6. Load behavior: all-or-nothing and rerun-safety

Every load runs inside a single transaction (`run_qasmbench_gold`'s
`with connections.postgres_connection(settings) as connection:` block,
which commits on a clean exit and rolls back on any exception):

1. Ensure the schema exists (`CREATE TABLE IF NOT EXISTS` — safe to repeat)
2. `TRUNCATE TABLE gold_qasmbench_circuit RESTART IDENTITY CASCADE` — clears
   all four tables in one statement, since the other three all reference
   `gold_qasmbench_circuit`
3. Bulk-insert circuit rows, then register rows, then stabilizer-check rows,
   then conditional-correction rows
4. Commit only if every step above succeeded

If any step raises an exception, the caller rolls back the entire
transaction — including the earlier `TRUNCATE` — so a failed load leaves the
previous complete Gold data untouched, never a half-loaded state. This is
the exact same strategy google_qec uses (Section 6 of its own report).

Because every load truncates before inserting, running the load twice in a
row produces identical row counts, not duplicates.

---

## 7. Row-count reconciliation

`row_counts.json` records what Silver had and what Gold actually loaded side
by side, with a `match` flag where the two are expected to be equal — the
`register` entry has no `silver_rows`/`match` pair, since registers are
derived (decomposed from `circuit.parquet`) rather than a 1:1 copy of a
Silver row count:

```json
{
  "circuit": { "silver_rows": 6, "gold_loaded": 6, "match": true },
  "register": { "gold_loaded": 16 },
  "stabilizer_check": { "silver_rows": 4, "gold_loaded": 4, "match": true },
  "conditional_correction": { "silver_rows": 6, "gold_loaded": 6, "match": true }
}
```

(These are the real counts from the supplied course archive, verified by an
end-to-end run of `prepare_qasmbench_silver` + `run_qasmbench_gold`: 6
circuits × ~2.67 registers on average — `qec_sm_n5` declares 4 registers per
variant (`q`, `a`, `c`, `syn`), the other two benchmarks declare 2 each
(`q`, `c`) — giving 4+4+2+2+2+2 = 16 total.)

---

## 8. Test coverage

The Gold stage is covered by 21 tests in `test_gold_qasmbench.py`, plus 2
more in `test_load_postgres_run.py` for the combined `run()`, all run
against a real, isolated throwaway Postgres schema (created and dropped per
test — the real `gold_qasmbench_*` tables are never touched):

- Schema creation (all four tables exist)
- All three foreign-key relationships reject orphaned rows
  (`gold_qasmbench_register`, `gold_qasmbench_stabilizer_check`,
  `gold_qasmbench_conditional_correction`, each → `gold_qasmbench_circuit`)
- Every value check, both the reject case and (where relevant) the valid
  case: `qubit_count > 0`, `variant IN (...)` (reject and accept both
  values), `register_size > 0`, `register_kind IN (...)`,
  `cardinality(data_qubits) >= 1`, `condition_value >= 0`
- Both uniqueness rules: `(benchmark_name, variant)` and
  `source_record_id`, independently
- `source_record_id` round-trips correctly from a Silver row to its Gold row
  (traceability) on all three source tables
- Register decomposition is correct: one row per declared register, with
  the right `register_kind`/`register_name`/`register_size`
- Repeatability: loading twice produces identical row counts, not
  duplicates
- The `TRUNCATE ... CASCADE` actually clears all three dependent tables,
  even when the next load has zero circuits (not just relying on new rows
  happening to overwrite old ones)
- **All-or-nothing:** a successful load followed by a deliberately broken
  second load correctly rolls back, leaving the original data completely
  intact
- The combined `run()` calls both sources' loaders exactly once each and
  sums their `StageResult` counts correctly

All tests pass:
`cd starter && uv run pytest tests/test_gold_qasmbench.py tests/test_load_postgres_google.py tests/test_load_postgres_run.py`
(requires a reachable PostgreSQL — `make up` from the repo root). Verified
locally against a real `postgres:16.9-bookworm` container: 37/37 passed
(2026-09-28).

Beyond the fixture-based unit tests above, the full pipeline was also run
end-to-end against the real supplied course archive and a real Postgres
instance (`prepare_qasmbench_silver` → `run_qasmbench_gold`), confirming the
real counts: 6 circuits, 16 registers, 4 stabilizer checks, 6 conditional
corrections, 0 data issues — matching `docs/decision-log.md`'s per-benchmark
table exactly.

---

## 9. Requirements coverage

| Requirement | How it is met |
| --- | --- |
| State what one row represents, per table | Section 2 |
| Student-designed tables, not a copy of Silver | Register decomposition (Section 5) |
| Primary keys, foreign keys, value checks, indexes | Section 3 |
| Rules driven by the data, not generic checks | Section 3, "Data-driven checks" |
| All-or-nothing load | Section 6 |
| Repeated loads create no duplicates | Section 6 (truncate-then-insert) |
| Every row traceable to its Silver source | `source_record_id` on circuit, stabilizer_check, conditional_correction (Section 2); tested directly |
| Reconciled row counts, not just a single number | Section 7 |
| No invented cross-source relationship | Section 10 |
| Reproducible SQL analysis, with one query joining ≥ 3 Gold tables | Section 11 |

---

## 10. Explicitly rejected relationship

Gold for `qasmbench` does **not** join to `google_qec` or `qec_syndromes` at
the row level. This was investigated in detail before any Gold code was
written — see `docs/decision-log.md`'s "Rejected candidate relationship"
section: QASMBench's 5-qubit repetition-code circuits
(`qreg q[3]; qreg a[2]`) are not the same code family as Google's 17-qubit
distance-3 surface-code layout (`data_qubits: 9`, `measure_qubits: 8`), even
though both are informally "distance 3." This is an intentional,
evidence-based design decision, not an oversight — `assignment/brief.md`:
"shared vocabulary does not create a shared identifier."

---

## 11. Part I analysis question 3

`assignment/brief.md`'s third Part I analysis question is qasmbench's:
"How does the repetition-code circuit map data qubits to parity-check
ancillas, syndrome bits, and conditional corrections?" Answered by
[`query3_qasmbench_circuit_mapping.sql`](query3_qasmbench_circuit_mapping.sql),
a 4-table join (`gold_qasmbench_circuit`, `gold_qasmbench_stabilizer_check`,
`gold_qasmbench_register`, `gold_qasmbench_conditional_correction`) — this is
also the join across three or more Gold tables the brief requires from at
least one analysis query. Verified against the real course archive: ancilla
`a[0]` checks `q[0]`/`q[1]` into `syn[0]`, `a[1]` checks `q[1]`/`q[2]` into
`syn[1]`, and every non-zero 2-bit syndrome value (1, 2, 3) has a matching
correction — value 3 (both checks fired) correctly targets `q[1]`, the qubit
shared by both checks. Source and transpiled variants agree row-for-row. See
the SQL file for the full result table and interpretation.

---

## 12. Limitations

- Only qasmbench's and google_qec's Gold tables exist so far; syndromes Gold
  loading is separate, not yet implemented in `load_postgres.py`.
- The load is a full truncate-then-reload of the whole qasmbench slice
  rather than a row-level upsert; this is a deliberate choice for this
  dataset's size (Section 6), not a general Gold-loading strategy.
- Which Postgres schema the whole Gold model should live in (`public`, a
  dedicated `gold` schema, or similar) is still open — see Section 4.
