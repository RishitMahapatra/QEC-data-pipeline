# Discovery Decision Log

This is the discovery decision log that is requested in the assignment, which answers each "questions to investigate" from `assignment/data-sources.md`.
Each answer gives the conclusion and points to the code, test or query that proves it, so the answer can be checked. The design
built and made on these findings is explained in the Part I report,
[reports/part1.md](../reports/part1.md).

---

## qec_syndromes

### Does the documented schema match the actual header? — *Stage: Bronze*
Yes, enforced by code, not just sampled. `register_sources.py`'s
`_check_syndrome_structure` reads the header line of all 7 CSVs and raises
`BronzeError` on any mismatch against `labels,syndromes,quantity`. This runs
every time the Bronze stage runs, so it can't silently regress.

### Does every parsed sequence have the same round/check shape and binary domain? — *Stage: Silver (shape/domain check in `prepare_data.py`)*
Yes, and it's confirmed across all 7 files, not just row 0. `build_syndromes`
rejects any row whose parsed `syndromes` value isn't exactly 4 tuples of 4
binary values (`syndromes.shape`). On the real data,
`starter/results/part1/data_issues.parquet` is empty, so there are 0
`syndromes.shape` issues across the whole dataset.
`test_malformed_tuple_is_rejected_into_issues` proves that the rule rejects
a badly shaped row.

### Do quantities reconcile to the filename sample count? — *Stage: Silver (positive-quantity + 10M reconciliation check)*
Yes, we checked all 7 files, not just one. Each file's accepted `quantity`
sums to **exactly 10,000,000**, matching its `nb-10M` filename hint, so the
dataset totals 70,000,000. No `syndromes.weighted_total_reconciliation`
warnings are raised on the supplied data. The same totals are checked
again from Silver to Gold in `row_counts.json` (`quantity_by_experiment`),
and they appear in the output of `query1_weighted_syndrome_by_fault_rate.sql`
(each fault rate's two label rows sum to 10M). See part1.md section 5.

### Can the same syndrome occur under both logical-error labels? — *Stage: Silver (must accept both rows as valid, not dedupe)*
Yes, it was confirmed against the real Gold data, not just a synthetic test
case. `test_same_syndrome_both_labels_are_both_kept` proves the code never
unduplicates a collision. By querying the real dataset
(`SELECT syndrome_bits FROM gold_syndrome_observation GROUP BY syndrome_bits HAVING COUNT(DISTINCT logical_error_label) > 1`), it shows that 18,676 distinct patterns occur under both labels across the 7 files (which is common and not a rare edge case). Both rows are correctly retained for each one, not deduped.

### What is the correct key if syndrome alone is not unique? — *Stage: Silver (`source_record_id` design for `syndrome_observation.parquet`)*
The key is the combination (experiment, syndrome, label). A syndrome on its own repeats across files and can show up with both labels, but this combination appears only once.
Silver: source_record_id = sha256(filename:row_index). Each row is identified by where it sits in its source file, which stays stable across reruns.
Gold: observation_id = sha256(experiment_id:syndrome_bits:label), backed by UNIQUE (experiment_id, syndrome_bits, logical_error_label) See
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

### Why must a model split keep an entire physical-fault-rate group together? — *Stage: ML export (course split), belongs to Part II*
Because the same syndrome patterns occur in many fault-rate files. A
row-level random split would put the same input in both train and test, so
the test score would mostly measure memorisation. Gold shows how much the
files overlap:

- 14,187 of the 31,941 distinct patterns occur in more than one fault-rate
  file.
- Of the 14,643 distinct patterns in the test file (fault rate 0.005),
  **14,168 (97 %) also occur in the training files**. This was measured
  with a join over `gold_syndrome_observation` and `gold_syndrome_experiment`.

Each file (one noise level) goes entirely into one split: 0.0005 → validation, 0.005 → test, the rest → train (53,304 / 1,407 / 20,887 rows). This way the test measures how the model does at a noise level it never trained on. The same syndrome pattern can appear in train and test, but with different label counts and weights, because it comes from a different experiment.

---

## google_qec


### Do filenames and properties agree? — *Stage: Silver (compare dirname-encoded values against `properties.yml` fields while parsing)*
Yes, for all 5 experiments. `prepare_data.py`'s `check_google_experiment_name`
parses `(basis, distance, rounds, row, col)` from the folder name
(`surface_code_b<basis>_d<distance>_r<rounds>_center_<row>_<col>`). It
compares that with `properties.yml`'s `basis`, `distance`, `rounds`,
`center_data_qubit_row` and `center_data_qubit_col`, and any difference stops
the run. On the supplied data all 5 agree:

| Folder | basis | d | rounds | center |
| --- | --- | ---: | ---: | --- |
| `surface_code_bX_d3_r25_center_3_5` | X | 3 | 25 | (3, 5) |
| `surface_code_bX_d3_r25_center_5_3` | X | 3 | 25 | (5, 3) |
| `surface_code_bX_d3_r25_center_5_7` | X | 3 | 25 | (5, 7) |
| `surface_code_bX_d3_r25_center_7_5` | X | 3 | 25 | (7, 5) |
| `surface_code_bX_d5_r25_center_5_5` | X | 5 | 25 | (5, 5) |

Gold also enforces `UNIQUE (basis, distance, rounds, center_row, center_col)`,
so two experiments can never claim the same configuration.

### Does every companion file exist? — *Stage: Bronze*
Yes, it is enforced by code. `register_sources.py`'s
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
means that the shots in that file no longer line up with the other files. In each
record, bits are little-endian (bit *i* is `(byte[i // 8] >> (i % 8)) & 1`).
The unused high bits of the last byte must be zero: `dirty_padding_shots`
excludes and logs any shot that breaks this. On the supplied data, no file
has the wrong size and no shot has non-zero padding. The corresponding tests:
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

### Beyond the packed per-shot bits the Google `ml` table needs, what should Gold do with detector events: one row per fired detector, or per-position summaries? Which supports the required analyses, and at what size? — *Stage: Gold (relational design decision)*
**Neither.** Gold keeps the packed bits (one `BYTEA` value per shot, needed
by the ML table) plus a per-shot `detector_event_count`. No required analysis
looks at single detectors, so neither alternative is built. Measured on the
real Gold data: one row per fired detector would be 10,446,925 rows and
1,439 MB, against 103 MB for the whole shot table; per-position summaries
would be only 1,400 rows (136 kB), but they lose the link to individual
shots. The full comparison and the cost of the choice are in part1.md
section 6.5.

### Which supplied decoders make different mistakes, and can a simple combined model use that disagreement to predict the actual label? — *Stage: Part II (Task B)*
**They make partly different mistakes, but a linear combination cannot use
that: it only ties the best single decoder.** The four decoders disagree on
55.4% (d=3) and 64.8% (d=5) of test shots, but all four are wrong together
on 14.4% and 10.9%, which no combination of their answers can fix. The
five-input combined model reaches 0.3977 at both distances, against 0.3967
(d=3) and 0.3952 (d=5) for the best decoder, tensor network contraction.
The full ranking, the overlap per decoder pair and the combined model's
coefficients are in part2.md, Task B.

### Why should distances three and five use separate raw-detector models? — *Stage: Part II (feature-width design)*
The inputs have different shapes and meanings:

- A d=3 shot has 200 detector bits (25 bytes); a d=5 shot has 600 (75
  bytes). A single flat feature vector cannot hold both without padding.
- Even with padding, bit *i* means a different detector at d=3 than at d=5,
  because the codes have different layouts. Postion *i* would have no
  shared meaning across distances.

The ML export's contract check therefore keeps one feature width per
distance. In Part II Task C , we train the MLP only on d=3 (`shot_index < 12_500`).

---

## qasmbench

This is the QASM parser feeding for items below (Silver stage). (File-existence for these 6 files is
already Bronze-checked by `_check_qasmbench_structure`; content parsing is
not.)

This is the must list of the things we had to extract:

- [x] stable member locator and file hash — `source_record_id` is
      `qasmbench:<member path>`; `source_trace.parquet` records the member and
      the SHA-256 of the Bronze zip
- [x] benchmark and source/transpiled variant — from the member's folder name
      and the `_transpiled.qasm` suffix; `circuit_id` = `<benchmark>:<variant>`
- [x] declared quantum/classical register sizes — `register_declarations`
      (JSON of qregs/cregs with sizes, in declaration order), plus
      `qubit_count`, the total of all declared qubits
- [x] operation, measurement, and two-qubit counts — the executed operations
      are counted as `measurement_count` and `two_qubit_gate_count`, as the
      Silver contract defines them. Custom gates are decomposed first (a gate
      body is a definition, counted only where it is invoked), and
      register-wide statements count once per qubit
- [x] parity-check ancilla, data participants, and syndrome bit where explicit
- [x] syndrome-controlled correction value, gate, and target

All 6 files parsed (3 benchmarks × source/transpiled):

| Benchmark | Registers | 2-qubit gates | Measurements | Checks | Corrections |
| --- | --- | ---: | ---: | ---: | ---: |
| `error_correctiond3_n5` | `q[5]`, `c[5]` | 49 | 5 | 0 | 0 |
| `qec_en_n5` | `q[5]`, `c[5]` | 10 | 5 | 0 | 0 |
| `qec_sm_n5` | `q[3]`, `a[2]`, `c[3]`, `syn[2]` | 4 | 5 | 2 | 3 |

Source and transpiled variants give the same numbers.

**Decision: what counts as an explicit parity check.** A check is recorded
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

**Decision: conditional operations.** Each single-qubit gate inside an
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

## Gold design

The Gold design for all three sources (tables, keys, constraints, design
decisions and the all-or-nothing load) is in
[reports/part1.md](../reports/part1.md), section 6. None of the QASMBench
Gold tables reference the `google_qec` or `qec_syndromes` tables; the
evidence is under "Rejected candidate relationship" below.

---

## Rejected candidate relationship

We investigated joining the QASMBench circuits to the distance-3 Google
experiments and the `d-3` syndrome files on the shared "distance 3" label,
and rejected it. The evidence and decision are in
[reports/part1.md](../reports/part1.md), section 7.
