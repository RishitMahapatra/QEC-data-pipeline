# Part I — discovery decision log

This is the discovery decision log that `assignment/getting-started.md` asks
for, which answers each "questions to investigate" from `assignment/data-sources.md`.
Each answer gives the conclusion and points to the code, test or query that
proves it, so the answer can be checked rather than just believed. The design
built on these findings is explained in the Part I report,
[reports/part1.md](../reports/part1.md).

---

## qec_syndromes

Questions from `assignment/data-sources.md`:

### Does the documented schema match the actual header? — *Stage: Bronze (done)*
Yes, enforced by code, not just sampled. `register_sources.py`'s
`_check_syndrome_structure` reads the header line of all 7 CSVs and raises
`BronzeError` on any mismatch against `labels,syndromes,quantity`. This runs
every time the Bronze stage runs, so it can't silently regress.

### Does every parsed sequence have the same round/check shape and binary domain? — *Stage: Silver (shape/domain check in `prepare_data.py`)*
Done — confirmed across all 7 files, not just row 0. `build_syndromes`
rejects any row whose parsed `syndromes` value isn't exactly 4 tuples of 4
binary values (`syndromes.shape`); running it against the real Bronze zip
(`test_accepted_quantity_reconciles_to_filename_hint_on_real_bronze_zip`)
produces 0 `syndromes.shape` issues across the whole dataset.

### Do quantities reconcile to the filename sample count? — *Stage: Silver (positive-quantity + 10M reconciliation check)*
Yes. We checked all 7 files, not just one. Each file's accepted `quantity`
sums to **exactly 10,000,000**, matching its `nb-10M` filename hint, so the
dataset totals 70,000,000. No `syndromes.weighted_total_reconciliation`
warnings are raised on the supplied data. The same totals are checked
again from Silver to Gold in `row_counts.json` (`quantity_by_experiment`),
and they appear in the output of `query1_weighted_syndrome_by_fault_rate.sql`
(each fault rate's two label rows sum to 10M). See part1.md section 5.

### Can the same syndrome occur under both logical-error labels? — *Stage: Silver (must accept both rows as valid, not dedupe)*
Done. Yes — confirmed against the real Gold data, not just a synthetic test
case. `test_same_syndrome_both_labels_are_both_kept` proves the code never
dedupes a collision; querying the real dataset
(`SELECT syndrome_bits FROM gold_syndrome_observation GROUP BY syndrome_bits HAVING COUNT(DISTINCT logical_error_label) > 1`)
shows **18,676 distinct patterns** occur under both labels across the 7
files — this is common, not a rare edge case. Both rows are correctly
retained for each one, not deduped.

### What is the correct key if syndrome alone is not unique? — *Stage: Silver (`source_record_id` design for `syndrome_observation.parquet`)*
Resolved, differently than the original candidate. Silver's
`source_record_id` is `sha256(filename:row_index)` — a hash of the row's
*position* in its source file, not a hash of its content
`(experiment_id, syndromes, labels)`. This is simpler and still fully stable
(same file + same row position -> same id on every rerun), and doesn't
require re-hashing the full syndrome value. The content-based composite key
idea was used instead at the Gold layer: `gold_syndrome_observation.observation_id`
is `sha256(experiment_id:syndrome_bits:label)`, matching this table's
`UNIQUE (experiment_id, syndrome_bits, logical_error_label)` constraint. See
part1.md section 6.4.

### Which aggregates must use `quantity` as a weight? — *Stage: Gold/ML (Silver only carries `quantity` through unchanged)*
Every aggregate that counts physical observations must use `quantity` as a
weight, because a single row can stand for up to millions of observations.
This covers frequencies, label shares, and error rates per fault rate or per
pattern. Counting rows instead would measure how many *distinct* (pattern,
label) pairs exist, not how often they happen.

- **Gold analysis:** `starter/results/part1/analysis/query1_weighted_syndrome_by_fault_rate.sql`
  computes `SUM(quantity) GROUP BY (physical_fault_rate, logical_error_label)`.
  The weighted share of logical errors rises from 0.02 % at fault rate
  0.00001 to 18.7 % at 0.01.
- **ML hand-off:** the export carries `quantity` into
  `ml_syndrome_decoder_example.sample_weight` (Σ = 70,000,000). Part II Task
  A must use it in every evaluation metric and, where the model supports it,
  in fitting (`assignment/part-2-ai-ml.md`).

### Why must a model split keep an entire physical-fault-rate group together? — *Stage: ML export (course split) / Part II*
Because the same syndrome patterns occur in many fault-rate files. A
row-level random split would put the same input in both train and test, so
the test score would mostly measure memorisation. Gold shows how much the
files overlap:

- 14,187 of the 31,941 distinct patterns occur in more than one fault-rate
  file.
- Of the 14,643 distinct patterns in the test file (fault rate 0.005),
  **14,168 (97 %) also occur in the training files**. This was measured
  with a join over `gold_syndrome_observation` and `gold_syndrome_experiment`.

Keeping each file whole also means the test answers a meaningful question:
*how does a model trained at other noise levels do at an unseen noise
level?* The course helper `ml.syndrome_data_split` assigns whole files:
0.0005 → validation, 0.005 → test, every other rate → train. The export
applies it per row from `physical_fault_rate`, so no file is ever split.
This gives 53,304 / 1,407 / 20,887 rows. A pattern seen in training can
still appear in the test file, but with that file's own label frequencies
and weights.

---

## google_qec

Questions from `assignment/data-sources.md`:

### Do filenames and properties agree? — *Stage: Silver (compare dirname-encoded values against `properties.yml` fields while parsing)*
Yes, for all 5 experiments. `prepare_data.py`'s `check_google_experiment_name`
parses `(basis, distance, rounds, row, col)` from the folder name
(`surface_code_b<basis>_d<distance>_r<rounds>_center_<row>_<col>`). It
compares that with `properties.yml`'s `basis`, `distance`, `rounds`,
`center_data_qubit_row` and `center_data_qubit_col`, and any difference stops
the run (`GoogleSilverError`). On the supplied data all 5 agree:

| Folder | basis | d | rounds | center |
| --- | --- | ---: | ---: | --- |
| `surface_code_bX_d3_r25_center_3_5` | X | 3 | 25 | (3, 5) |
| `surface_code_bX_d3_r25_center_5_3` | X | 3 | 25 | (5, 3) |
| `surface_code_bX_d3_r25_center_5_7` | X | 3 | 25 | (5, 7) |
| `surface_code_bX_d3_r25_center_7_5` | X | 3 | 25 | (7, 5) |
| `surface_code_bX_d5_r25_center_5_5` | X | 5 | 25 | (5, 5) |

Gold also enforces `UNIQUE (basis, distance, rounds, center_row, center_col)`,
so two experiments can never claim the same configuration.

### Does every companion file exist? — *Stage: Bronze (done)*
Yes — enforced by code, not just sampled. `register_sources.py`'s
`_check_google_structure` confirms exactly 5 experiment directories and all
15 documented companion files per directory, raising `BronzeError` on any
gap. Runs every Bronze execution.

### Does every binary file length match shots times byte-aligned bits per shot? — *Stage: Silver (`b8` decode/reconciliation check)*
Yes, all 15 `b8` files (3 per experiment). The bit counts come from
`properties.yml`'s real keys: `circuit_measurements`, `circuit_detectors` and
`circuit_sweep_bits`. These are not the `measurement_count`/`detector_count`
names we first guessed. Each shot is padded to whole bytes, so a file must be
exactly `shots × ceil(bits / 8)` bytes:

| Experiment | File | Bits per shot | Bytes per shot | Expected = actual size |
| --- | --- | ---: | ---: | ---: |
| d=3 (each of 4) | `measurements.b8` | 209 | 27 | 1,350,000 |
| d=3 (each of 4) | `detection_events.b8` | 200 | 25 | 1,250,000 |
| d=3 (each of 4) | `sweep.b8` | 9 | 2 | 100,000 |
| d=5 | `measurements.b8` | 625 | 79 | 3,950,000 |
| d=5 | `detection_events.b8` | 600 | 75 | 3,750,000 |
| d=5 | `sweep.b8` | 25 | 4 | 200,000 |

`check_b8_size` stops the run on any size mismatch, because a wrong length
means the shots in that file no longer line up with the other files. In each
record, bits are little-endian (bit *i* is `(byte[i // 8] >> (i % 8)) & 1`).
The unused high bits of the last byte must be zero: `dirty_padding_shots`
excludes and logs any shot that breaks this. On the supplied data, no file
has the wrong size and no shot has non-zero padding. Tests:
`test_packed_bits_are_little_endian_within_each_byte`,
`test_padding_is_the_high_bits_of_the_last_byte`,
`test_shot_with_dirty_padding_is_excluded_and_reported`.

### Does every `01` file contain one binary line per shot? — *Stage: Silver (line-count-vs-`shots` check)*
Yes, all 25 `01` files (5 per experiment: `obs_flips_actual.01` and the four
`obs_flips_predicted_by_<decoder>.01`) have exactly 50,000 lines, each `0` or
`1`. `read_01_flags` stops the run if a file has the wrong number of lines or
a value other than 0/1. Line *i + 1* of each file belongs to the same shot as
record *i* of the `b8` files. This alignment is recorded per shot in
`source_trace.parquet`, as 8 trace rows: `shot_index=i` for `b8` files and
`line=i+1` for `01` files.

### What distinguishes an actual logical flip from a decoder mistake? — *Stage: Silver (keep as separate typed columns in `shot.parquet`)*
They are different facts and are stored separately at every layer:

| Concept | Meaning | Where it is stored |
| --- | --- | --- |
| measurement | raw qubit measurement results | `measurement_bits` (from `measurements.b8`) |
| detector event | a change in a parity-check outcome between rounds | `detector_bits`, `detector_event_count` |
| actual logical flip | ground truth: did the logical observable flip? | `actual_observable_flip` (from `obs_flips_actual.01`) |
| decoder prediction | what one decoder *guessed* about that flip | Silver: four `*_prediction` columns; Gold: `gold_google_decoder_prediction.predicted_flip` |
| decoder mistake | the guess disagrees with the actual flip | Gold only: `gold_google_decoder_prediction.was_wrong = (predicted_flip <> actual_observable_flip)` |

A mistake is derived, not observed, so Silver does not store it. Gold
computes it once per (shot, decoder). A flip that a decoder predicts
correctly is not a mistake, and a decoder can be wrong whether or not a flip
happened. In the Google trace example (part1.md section 9.2), the actual flip
is `true`: belief matching and tensor network predict `true` (correct), while
correlated matching and PyMatching predict `false` (`was_wrong = true`).

### What should Gold do with detector events: one row per fired detector, or per-position summaries? — *Stage: Gold (relational design decision)*
Store them packed: **one `BYTEA` value per shot**, plus a per-shot
`detector_event_count`. We measured both options before choosing:

- Packed: 200,000 × 25 + 50,000 × 75 = **8,750,000 bytes** of detector data.
  This was confirmed in Gold with `SUM(octet_length(detector_bits))`. The
  whole `gold_google_shot` table is 103 MB (`pg_total_relation_size`).
- One row per detector: 200 × 200,000 + 600 × 50,000 = **70,000,000 rows**,
  which is 280× more rows.

The required analyses only need `detector_event_count`, and the ML table
needs the packed bytes, which the export copies from Gold unchanged (tested
byte for byte against Silver). The cost is that SQL cannot filter on one
detector; the bits must be unpacked in Python. See part1.md section 6.5.

### Which supplied decoders make different mistakes, and can a simple combined model use that disagreement? — *Stage: Part II (Task B)*
Part I gives a first answer: the decoders' error *rates* differ in a stable
way (query 2, part1.md section 10). Tensor network contraction and belief
matching are best, then correlated matching, then PyMatching. Individual
shots show disagreement too: in the trace example, two decoders are right
and two are wrong on the same shot. Whether a linear combination of the four
predictions can use this is Part II Task B, and the answer will be added
here when Part II is done.

### Why should distances three and five use separate raw-detector models? — *Stage: Part II (feature-width design)*
The inputs have different shapes and meanings:

- A d=3 shot has 200 detector bits (25 bytes); a d=5 shot has 600 (75
  bytes). A single flat feature vector cannot hold both without padding.
- Even with padding, bit *i* means a different detector at d=3 than at d=5,
  because the codes have different layouts. Position *i* would have no
  shared meaning across distances.

The ML export's contract check therefore keeps one feature width per
distance. Part II Task C trains its MLP only on d=3 (`shot_index < 12_500`).

---

## qasmbench

*Stage: Silver for all items below — this is the QASM parser feeding
`circuit.parquet`, `stabilizer_check.parquet`, and
`conditional_correction.parquet`. (File-existence for these 6 files is
already Bronze-checked by `_check_qasmbench_structure`; content parsing is
not.)*

No explicit question list in `data-sources.md`; it instead lists what to
extract at minimum:

- [x] stable member locator and file hash — `source_record_id` is
      `qasmbench:<member path>`; `source_trace.parquet` records the member and
      the SHA-256 of the Bronze zip
- [x] benchmark and source/transpiled variant — from the member's folder name
      and the `_transpiled.qasm` suffix; `circuit_id` = `<benchmark>:<variant>`
- [x] declared quantum/classical register sizes — `register_declarations`
      (JSON of qregs/cregs with sizes, in declaration order)
- [x] measurement and two-qubit counts, expanded — custom gates are
      decomposed first (a gate body is a definition, counted only where it is
      invoked), and register-wide statements count once per qubit
- [x] parity-check ancilla, data participants, and syndrome bit where explicit
- [x] syndrome-controlled correction value, gate, and target

All 6 files parsed (3 benchmarks × source/transpiled):

| Benchmark | Registers | 2-qubit gates | Measurements | Checks | Corrections |
| --- | --- | ---: | ---: | ---: | ---: |
| `error_correctiond3_n5` | `q[5]`, `c[5]` | 49 | 5 | 0 | 0 |
| `qec_en_n5` | `q[5]`, `c[5]` | 10 | 5 | 0 | 0 |
| `qec_sm_n5` | `q[3]`, `a[2]`, `c[3]`, `syn[2]` | 4 | 5 | 2 | 3 |

Source and transpiled variants give the same numbers.

**Decision — what counts as an explicit parity check.** A check is recorded
when data qubits act as CNOT controls onto one ancilla qubit that is only ever
a CNOT target (no single-qubit gates on it), and that ancilla is then measured
into a classical bit. That is the syndrome bit. In `qec_sm_n5`, `a[0]` checks
`q[0],q[1]` into `syn[0]` and `a[1]` checks `q[1],q[2]` into `syn[1]`.
`error_correctiond3_n5` and `qec_en_n5` produce no check rows on purpose: they
have no separate ancilla register, and the qubit collecting the parity
(`q[2]`) also gets single-qubit gates (H, and T in `qec_en_n5`) between the
CNOTs, and is measured together with the data qubits into one register
`c[5]`. Nothing in the file
says which classical bit is a syndrome bit. `data-sources.md` asks for checks
"where explicit", so we do not guess a role for these qubits.

**Decision — conditional operations.** Each single-qubit gate inside an
`if(reg==value)` is one `conditional_correction` row; `if(...) x q;` on a whole
register gives one row per qubit. Gates inside an `if` block are not added to
`measurement_count` / `two_qubit_gate_count`, because they only run when the
condition holds. None of the 6 files has a conditional two-qubit gate or
measurement, so this choice does not change any current count.

**Rejected files.** A member that cannot be decoded as UTF-8 or parsed as
OpenQASM 2 is written to `data_issues.parquet` (`qasmbench.parse_failed`, the
member path as value, the parser message as reason) and adds no rows to any
of the three tables. Its trace row is kept so the issue still traces to
Bronze. `row_counts.json` reconciles at circuit level (read = accepted +
rejected); checks and corrections come only from accepted circuits.

---

## qasmbench Gold design

*Stage: Gold (`load_postgres.py`, `load_qasmbench_gold` / `_load_qasmbench`).
Loads the three QASMBench Silver tables into four Postgres tables, scoped to
QASMBench only for its own schema/rows — but as of the merge with
`google_qec`'s and `qec_syndromes`' Gold code (from the team's
`rishit_mahapatra` and `dari` branches), qasmbench loads inside the same
single shared transaction as every other source, in `run()`. All three
sources' code lives in one `load_postgres.py`, and qasmbench follows the
same two-tier shape google_qec established: `load_qasmbench_gold(conn,
settings)` is a connection-scoped loader (truncate + insert + return counts
and hashes, no commit of its own), and `_load_qasmbench(conn, settings)`
wraps it — runs the schema DDL, calls the loader, then re-queries the actual
Gold tables and raises if they disagree with what the loader returned.*

**Decision — table names and shared `gold` schema (team decision, resolved
across all three sources together).** Tables are `gold_qasmbench_circuit`,
`gold_qasmbench_register`, `gold_qasmbench_stabilizer_check`,
`gold_qasmbench_conditional_correction`. `run()` does
`CREATE SCHEMA IF NOT EXISTS gold` and `SET search_path TO gold` once, before
loading any source, so every source's Gold tables — `gold_google_*`,
`syndrome_*`, and `gold_qasmbench_*` alike — live in the same dedicated
`gold` schema, not `public`. An earlier draft of this design put qasmbench's
tables in their own `qec` schema; that was dropped once a teammate flagged
during review that the team should pick one schema strategy together rather
than each source doing its own thing, and `google_qec`'s and `qec_syndromes`'
code (independently arriving at the same `gold` schema) was already the
majority direction.

**Decision — re-query the actual Gold table instead of trusting the
loader's own count (also from that same review).** An earlier version of
this code wrote `row_counts.json`'s `"gold_loaded"` field from the *same*
Python variable as `"silver_rows"`, making `"match"` trivially always
`True` — it never actually checked the database. `_load_qasmbench` now runs
`SELECT COUNT(*)` against each of the four real tables after loading and
compares that independent count to what `load_qasmbench_gold` returned;
any disagreement raises `RuntimeError` *inside* the still-open transaction,
which rolls back not just qasmbench's rows but every source loaded earlier
in the same `run()` call (google_qec, syndromes) — proven by
`test_load_qasmbench_detects_row_count_mismatch_and_raises` and
`test_run_rolls_back_google_and_syndromes_when_qasmbench_load_fails`.

**Decision — normalize `register_declarations` instead of copying it.**
Silver's `circuit.parquet` keeps `register_declarations` as one JSON-text
column (a nested list of qreg/creg name+size pairs). Gold decomposes it into
its own table, `gold_qasmbench_register` (one row per declared register, FK
to `gold_qasmbench_circuit`), instead of also carrying a JSONB copy of the
same column. Two representations of the same fact with nothing to keep them
in sync is worse than one; the brief says Gold must not "merely copy each
Silver table into PostgreSQL", and turning an opaque JSON blob into typed,
constrained (`register_size > 0`, `register_kind IN ('qreg','creg')`) rows is
exactly that kind of value-add. Only 4 registers per circuit variant (q, a,
c, syn for the parity-check benchmark) — decomposing costs one small table,
not real complexity.

**Decision — keep `data_qubits` as a native `text[]` column, not a junction
table.** `stabilizer_check.parquet`'s `data_qubits` is a small (2 elements),
fixed-shape, ordered attribute of one check row, not an independent
many-to-many relationship worth its own table — across the whole QASMBench
archive it's ~8 total (qubit, check) pairs.
`gold_qasmbench_stabilizer_check.data_qubits` is `text[]` with
`CHECK (cardinality(data_qubits) >= 1)`. Postgres arrays are queryable
(`= ANY(data_qubits)`, `&&`) without the overhead of a fifth table for this
data's size. A stricter 3NF read could argue for a junction table instead;
the size/shape of this specific data is the reason this project didn't take
that path. (The constraint uses `cardinality()`, not `array_length(..., 1)`:
`array_length` of an empty array returns `NULL`, and a `CHECK` expression
that evaluates to `NULL` — as opposed to `FALSE` — passes, so an
`array_length`-based check silently let an empty `data_qubits` array through
until a test caught it. `cardinality()` returns `0` for an empty array, so
the comparison actually evaluates to `FALSE`.)

**Decision — full truncate-then-reload, not upsert, inside the shared
cross-source transaction.** Each call to `load_qasmbench_gold` does
`TRUNCATE TABLE gold_qasmbench_circuit RESTART IDENTITY CASCADE` (which
clears `gold_qasmbench_register`, `gold_qasmbench_stabilizer_check`, and
`gold_qasmbench_conditional_correction` too, since they all reference the
circuit table), then re-inserts every row. Unlike the design's first draft,
this no longer opens its own connection/transaction — `run()` opens one
connection, calls `_load_google`, `_load_syndromes`, then `_load_qasmbench`
in sequence on it, and only commits once all three succeed (`conn.commit()`
after the third call; any exception anywhere rolls back everything and the
connection is closed in a `finally`). QASMBench's Gold slice is small
(6 circuits, ~4 checks, ~6 corrections) and derived deterministically from
an unchanged Bronze archive, so a full rebuild each run is simpler to reason
about than a natural-key upsert and still satisfies "repeated runs on
unchanged inputs must leave stable identifiers and must not add duplicate
business records" — a rerun always lands on the same `circuit_id`s and row
counts. This mirrors google_qec's own truncate-then-reload strategy exactly.

**Cross-source isolation.** None of these four tables reference
`google_qec` or `qec_syndromes` Gold tables. That decision, and the evidence
for it, is already recorded below under "Rejected candidate relationship" —
not repeated here.

---

## Rejected candidate relationship

*Stage: Gold (this decides whether to create a cross-source relationship in
the relational model — Silver never joins sources at all).*

**Candidate:** join `qasmbench`'s `small/error_correctiond3_n5/` circuit to
the distance-3 `google_qec` experiments and/or the `d-3` `qec_syndromes`
files, on the shared "distance 3" label in their names.

**Evidence against:**
- `qec_sm_n5.qasm` (same family as `error_correctiond3_n5`) declares
  `qreg q[3]; qreg a[2]` — 5 qubits total, a repetition-code structure with a
  2-bit syndrome register.
- The distance-3 `google_qec` `properties.yml` reports `data_qubits: 9`,
  `measure_qubits: 8` — 17 qubits, a surface-code structure with
  `circuit_detectors: 200` detector bits over 25 rounds.
- Different qubit counts and different code families under the same "distance
  3" wording — per `brief.md`: "shared vocabulary does not create a shared
  identifier."

**Decision:** do not join QASMBench circuits to Google or syndrome
experiments by distance value or any other name-based match. Keep QASMBench
circuits as their own Gold entities, linked only to their own
parity-check/correction structure — consistent with `brief.md`: "QASMBench
cannot be joined row by row to the Google or simulated experiments... do not
invent a match."

Confirmed from the file itself: `error_correctiond3_n5.qasm` declares only
`qreg q[5]; creg c[5];`, so it has 5 qubits and no separate ancilla or
syndrome register. That is far from the 17-qubit distance-3 Google layout, so
the rejection stands.
