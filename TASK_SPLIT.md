# Task split — Silver, Gold, ML (5 people)

Bronze is done (checks implemented, evidence files correct, tests passing).
This is a proposed split for the remaining work, extending the 2–3 person
plan in `TEAM_GUIDE.md` section 7 to 5 people.

**Caution:** total estimated effort for Part I is only ~28–34 *team*-hours —
across 5 people that's ~6 hours each. Don't over-parallelize into isolated
silos; Gold especially needs shared understanding regardless of who writes
the code.

## P1 — Syndrome track

- Silver #1: `silver/qec_syndromes/syndrome_observation.parquet`
- Syndrome-side Gold entities
- `ml_syndrome_decoder_example` export
- Analysis query 1: weighted syndrome frequency/labels vs physical fault rate
- Part II Task A (weighted prior + weighted linear decoder)

Lightest-volume source (7 CSVs) — good fit for one person end-to-end.

## P2 — QASMBench track

- Silver #4–6: `circuit.parquet`, `stabilizer_check.parquet`,
  `conditional_correction.parquet`
- The QASM-parsing logic (register sizes, executed-op counts, parity checks,
  syndrome-controlled corrections)
- Analysis query 3: data qubit → ancilla → syndrome bit → correction mapping
- The rejected-relationship writeup/evidence (QASMBench vs Google/syndromes)
  — this person will know the circuit structure well enough to argue it

Self-contained — 3 circuits, no shot-level data.

## P3 — Google Silver track (heaviest single piece)

- Silver #2–3: `experiment.parquet`, `shot.parquet`
- `b8`/`01` decoding, byte-length reconciliation, alignment across companion
  files, for all 5 experiments

Comparable in effort to P1+P2 combined — dedicated person, possibly with P4
pairing in during crunch weeks.

## P4 — Google ML + Part II B/C track

- `ml_google_decoder_example` export
- Analysis query 2: decoder logical-error rate by distance and distance-3
  processor location
- Part II Task B (4-decoder comparison + linear combined decoder)
- Part II Task C (bounded raw-detector MLP)

Pairs naturally with P3 — both need deep familiarity with the same
shot-level data. P4 can start once P3's Silver tables stabilize.

## P5 — Gold + integration + tracing

- Gold DDL and the all-or-nothing `load_postgres.py`
- `source_trace.parquet` + `trace_examples.json` (inherently cross-source —
  one owner avoids three people writing conflicting tracing logic)
- Wiring `cli.py`'s `command_run` to orchestrate all four stages
- Driving the clean-environment rerun check

Becomes the de facto integrator once the others' Silver tables land.

## Explicitly shared, not siloed

- **Gold schema design itself** — one working session with all 5 before P5
  implements, since the brief requires a real relational model, not "copy
  each Silver table."
- **Both written reports** (Part I design report, Part II report) — each
  person writes the paragraph for their own area; P5 assembles them, given
  the integrator role.
- **Tests** for each person's own area stay with them; P5 owns the
  repeatability/idempotency tests since those cut across all stages.
