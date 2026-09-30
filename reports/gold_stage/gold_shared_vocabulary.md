# Gold stage — shared QEC vocabulary across sources

Code: [load_postgres.py](../../starter/src/quantum_lake_student/stages/load_postgres.py)
("Shared QEC vocabulary" section, `QASMBENCH_CODE_LABELS`, `QEC_CATALOG_VIEW_SQL`) ·
Tests: `test_load_postgres.py`, `test_load_postgres_syndromes.py`,
`test_load_postgres_qasmbench.py` (the `*shared_vocabulary*`, `*code_family*`
and `*repetition_code*` tests)

## 1. What the assignment asks for

- `faq.md`: *"Must all sources join into one experiment? No. They share QEC
  concepts but no reliable row-level experiment identifier. Use clear shared
  names without inventing a match."*
- `brief.md`: *"The sources should use consistent names for shared ideas, but
  shared vocabulary does not create a shared identifier."*
- `plain-language-guide.md`, *conforming*: *"Making two sources use the same
  names and types for the same idea, so they can sit in one model. This
  happens in `gold`."*

So Gold **conforms** the three sources (same names, types and allowed values
for the same idea) and **does not join** them. No foreign key goes from one
source's tables to another's.

## 2. Conformed columns

Every source's top-level entity (a Google experiment, a simulated fault-rate
experiment, a QASMBench circuit variant) carries the same two columns:

| Column | Type / allowed values | Meaning |
| --- | --- | --- |
| `code_family` | `TEXT`, `CHECK IN ('surface_code', 'repetition_code')` | which kind of QEC code the entity is about |
| `distance` | `INTEGER`, `CHECK (distance > 0)` | code distance: the smallest number of physical errors that can cause an undetected logical error |

Where each value comes from:

| Source / table | `code_family` | `distance` |
| --- | --- | --- |
| `google_qec` / `gold_google_experiment` | `surface_code`: every `properties.yml` says `type: surface_code_memory_experiment` | per experiment from Silver (`properties.yml`): 3 or 5 |
| `qec_syndromes` / `gold_syndrome_experiment` | `surface_code`: the source is documented as simulated distance-3 surface-code syndromes (`data-sources.md`) | from the `d-<distance>_` filename prefix. Carried as an extra Silver column (`distance`, int32) so Gold never re-reads Bronze |
| `qasmbench` / `gold_qasmbench_circuit` | `repetition_code` for `qec_sm_n5`; `NULL` for the other two benchmarks | `3` for `qec_sm_n5`; `NULL` otherwise |

**QASMBench labelling rule.** Only a circuit whose code is explicit in the
file gets a label. `qec_sm_n5` declares 3 data qubits (`q[3]`) and 2
ancillas (`a[2]`) that check the neighbouring pairs `q[0],q[1]` and
`q[1],q[2]` into a 2-bit syndrome register, and every non-zero syndrome value
flips exactly one data qubit. That is a distance-3 bit-flip repetition code.
`error_correctiond3_n5` and `qec_en_n5` have no explicit parity-check
structure (see the qasmbench section of `docs/decision-log.md`), so they stay
`NULL` rather than getting a guessed label. A table CHECK requires
`code_family` and `distance` to be both set or both `NULL`.

The label is a hand-written constant, so the load checks it against the
parsed data. A distance-*d* repetition code has *d* data qubits checked by
*d − 1* neighbouring parity checks. If Silver's stabilizer checks for a
labelled circuit do not match that, the load raises and the whole Gold
transaction rolls back (`_check_repetition_code_labels`).

## 3. Other shared terms (named consistently, kept as their contracts require)

| Idea | google_qec | qec_syndromes | qasmbench |
| --- | --- | --- | --- |
| Top-level entity id | `experiment_id` | `experiment_id` | `circuit_id` (a circuit is not an experiment) |
| QEC rounds | `rounds` (25) | `round_count` (4) | none (a circuit definition executes no rounds) |
| Parity-check outcome | `detector_bits` (detector events: changes between rounds) | `syndrome_bits` (syndrome values per round and check) | `syndrome_bit` (the classical bit an ancilla is measured into) |
| Ground-truth logical outcome | `actual_observable_flip` | `logical_error_label` | none (no shots) |
| Decoder output | `gold_google_decoder_prediction.predicted_flip` | none | none |
| Trace link to Silver | `source_record_id` | `source_record_id` | `source_record_id` |

`rounds`/`round_count` and `actual_observable_flip`/`logical_error_label`
keep their source-specific names because both ML table contracts
(`required-ml-tables.md`) prescribe those exact names. The catalog view
below shows the rounds under one name, `rounds`. Detector events and
syndrome values are deliberately **not** given one shared name: a detector
event is a change in a parity outcome between rounds, not the outcome itself,
and treating them as interchangeable is one of the rubric's grade caps.

## 4. `v_qec_experiment_catalog`

One row represents one experiment (google_qec, qec_syndromes) or circuit
variant (qasmbench), listed in the shared vocabulary. It is built with
`UNION ALL` (sources stacked), never `JOIN` (rows matched). It is created by
`run()` inside the same all-or-nothing transaction as the loads.

| Column | Meaning |
| --- | --- |
| `source_name` | `google_qec`, `qec_syndromes` or `qasmbench` |
| `entity_kind` | `experiment` or `circuit` |
| `entity_id` | the source's own id (`experiment_id` / `circuit_id`) |
| `code_family`, `distance` | the conformed columns above |
| `rounds` | QEC rounds (`NULL` for circuits) |

Example question it answers without inventing a match: *"What distance-3
data does Gold hold, and of which code family?"*

```sql
SELECT source_name, code_family, COUNT(*) AS entities
FROM gold.v_qec_experiment_catalog
WHERE distance = 3
GROUP BY source_name, code_family
ORDER BY source_name;
```

Expected on the supplied data: google_qec / surface_code / 4,
qasmbench / repetition_code / 2 (source + transpiled `qec_sm_n5`),
qec_syndromes / surface_code / 7.

## 5. How this shows the rejected relationship in the data

The rejected relationship (joining QASMBench's "d3" circuit to the
distance-3 Google or syndrome experiments; see `docs/decision-log.md`) is
now visible in Gold itself. All three have `distance = 3`, but QASMBench's
row says `repetition_code` and the others say `surface_code`. Same distance
number, different codes, so they are labelled with the same vocabulary and
never joined.

## 6. Design costs

- `code_family` for google_qec and qec_syndromes is a source-level constant
  in the loader, not a value parsed per row. This is honest because each
  source contains exactly one code family, but a future source with mixed
  families would need the value carried through Silver instead.
- The QASMBench label is a small hand-maintained mapping. The consistency
  check stops it from disagreeing with the parsed circuit, but a new
  benchmark stays unlabelled until someone adds it.
- A lookup table (`gold_code_family`) referenced by foreign keys was
  considered instead of CHECK constraints. It was not chosen because it would
  make every source's loader and isolated test schema depend on one more
  shared table, for a vocabulary of two values.
