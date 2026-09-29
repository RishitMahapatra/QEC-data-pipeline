# Part I — discovery decision log

Working notes from Phase 0 (investigation). Notebook cells produce the
evidence; this file records the conclusions and decisions that follow from
that evidence, per `assignment/getting-started.md` ("create a discovery
notebook and a plain Markdown decision log"). This feeds the final Part I
design report at submission time.

Answer each question with: the conclusion, and a pointer to which notebook
cell/output it's based on (so it stays traceable, not just asserted).

---

## qec_syndromes

Questions from `assignment/data-sources.md`:

### Does the documented schema match the actual header? — *Stage: Bronze (done)*
Yes — enforced by code, not just sampled. `register_sources.py`'s
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
Done — confirmed across all 7 files, not just one. Total accepted `quantity`
across the real dataset is ~70,000,000, within 2% of the combined `nb-10M`
filename hints (7 files × ~10M), and 0
`syndromes.weighted_total_reconciliation` warnings are raised on the
supplied data. See `reports/silver_stage/silver_stage_syndromes.md`.

### Can the same syndrome occur under both logical-error labels? — *Stage: Silver (must accept both rows as valid, not dedupe)*
Done. Yes — confirmed against the real Gold data, not just a synthetic test
case. `test_same_syndrome_both_labels_are_both_kept` proves the code never
dedupes a collision; querying the real dataset
(`SELECT syndrome_bits FROM syndrome_observation GROUP BY syndrome_bits HAVING COUNT(DISTINCT logical_error_label) > 1`)
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
idea was used instead at the Gold layer: `syndrome_observation.observation_id`
is `sha256(experiment_id:syndrome_bits:label)`, matching this table's
`UNIQUE (experiment_id, syndrome_bits, logical_error_label)` constraint. See
`reports/gold_stage/gold_stage_syndromes.md`, decision 4.

### Which aggregates must use `quantity` as a weight? — *Stage: Gold/ML (Silver only carries `quantity` through unchanged)*
Done for Gold's analysis query. `reports/gold_stage/query1_weighted_syndrome_by_fault_rate.sql`
computes exactly `sum(quantity) GROUP BY (physical_fault_rate, logical_error_label)`
— confirms the weighted, not row-count, logical-error share rises from
~0.02% at fault rate 0.00001 to ~18.6% at 0.01. Still open: which ML-side
(Part II) aggregations need the same weighting — that's Victor's stage, not
yet checked against this.

### Why must a model split keep an entire physical-fault-rate group together? — *Stage: Part II (ML split design, not a cleaning question)*
TODO

---

## google_qec

Questions from `assignment/data-sources.md`:

### Do filenames and properties agree? — *Stage: Silver (compare dirname-encoded values against `properties.yml` fields while parsing)*
TODO

### Does every companion file exist? — *Stage: Bronze (done)*
Yes — enforced by code, not just sampled. `register_sources.py`'s
`_check_google_structure` confirms exactly 5 experiment directories and all
15 documented companion files per directory, raising `BronzeError` on any
gap. Runs every Bronze execution.

### Does every binary file length match shots times byte-aligned bits per shot? — *Stage: Silver (`b8` decode/reconciliation check)*
TODO — `properties.yml` real keys: `shots`, `circuit_measurements`,
`circuit_detectors`, `circuit_sweep_bits` (not the guessed
`measurement_count`/`detector_count`). Checked `measurements.b8`,
`detection_events.b8`, `sweep.b8` for one experiment so far.

### Does every `01` file contain one binary line per shot? — *Stage: Silver (line-count-vs-`shots` check)*
TODO

### What distinguishes an actual logical flip from a decoder mistake? — *Stage: Silver (keep as separate typed columns in `shot.parquet`)*
TODO — `actual_observable_flip` is ground truth; a decoder "mistake" is one of
the four `*_prediction` columns disagreeing with it. Keep these as separate
concepts in Silver (per `silver-tables.md`), never collapse them.

### What should Gold do with detector events: one row per fired detector, or per-position summaries? — *Stage: Gold (relational design decision)*
TODO — decide and measure both approaches' size before choosing.

### Which supplied decoders make different mistakes, and can a simple combined model use that disagreement? — *Stage: Part II (Task B)*
Out of scope for Phase 0 — relevant to Part II Task B.

### Why should distances three and five use separate raw-detector models? — *Stage: Part II (feature-width design)*
TODO — note: feature width differs (`detector_count` is 200 for d=3 vs 600 for
d=5, per `required-ml-tables.md`), so a single flat feature vector can't cover
both.

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

## Rejected candidate relationship

*Stage: Gold (this decides whether to create a cross-source relationship in
the relational model — Silver never joins sources at all). The register-size
TODO at the bottom is still Phase 0/notebook work.*

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
