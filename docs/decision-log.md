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
TODO — confirmed for row 0 of `d-3_pfr-0.000010_nb-10M.csv` only so far.
Needs checking across all rows, all 7 files.

### Do quantities reconcile to the filename sample count? — *Stage: Silver (positive-quantity + 10M reconciliation check)*
TODO — confirmed `sum(quantity) == 10,000,000` for `d-3_pfr-0.000010_nb-10M.csv`
(matches `nb-10M`) only so far. Needs checking across all 7 files.

### Can the same syndrome occur under both logical-error labels? — *Stage: Silver (must accept both rows as valid, not dedupe)*
TODO — `d-3_pfr-0.000010_nb-10M.csv` showed 0/68 collisions (no syndrome seen
under both labels in that file). Must re-run across all 7 files, especially
higher fault-rate ones, before concluding either way.

### What is the correct key if syndrome alone is not unique? — *Stage: Silver (`source_record_id` design for `syndrome_observation.parquet`)*
TODO — candidate: `(experiment_id, syndromes, labels)`, where `experiment_id`
derives from the file (distance + physical fault rate). Verify this composite
is actually unique per file (no duplicate aggregate rows).

### Which aggregates must use `quantity` as a weight? — *Stage: Gold/ML (Silver only carries `quantity` through unchanged)*
TODO — e.g. logical-error rate per fault rate must be
`sum(quantity where label) / sum(quantity)`, not a row count.

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

- [ ] stable member locator and file hash
- [ ] benchmark and source/transpiled variant
- [ ] declared quantum/classical register sizes
- [ ] operation, measurement, and two-qubit counts (expanded, not just QASM
      statement lines)
- [ ] parity-check ancilla, data participants, and syndrome bit where explicit
- [ ] syndrome-controlled correction value, gate, and target

Sampled so far: `small/qec_sm_n5/qec_sm_n5.qasm` — 3 data qubits (`q[3]`), 2
ancillas (`a[2]`), 2-bit syndrome register, conditional X correction on
syndrome values 1/2/3. Still need `error_correctiond3_n5` and `qec_en_n5`
(source + transpiled variants of all three = 6 files total).

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

TODO: confirm by actually opening `error_correctiond3_n5.qasm` (not just
`qec_sm_n5.qasm`) and recording its real register sizes here.
