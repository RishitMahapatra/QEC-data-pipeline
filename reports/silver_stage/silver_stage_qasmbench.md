# Silver stage — `qasmbench`

This section explains the main decisions taken in the Silver stage for the
QASMBench QEC circuits and why they were taken.

Code: [prepare_data.py](../../starter/src/quantum_lake_student/stages/prepare_data.py) ·
Tests: [test_silver_qasmbench.py](../../starter/tests/test_silver_qasmbench.py)

## Input and output

- **Input:** the Bronze archive `qasmbench-qec.zip`, which holds 3 benchmark
  circuits (`error_correctiond3_n5`, `qec_en_n5`, `qec_sm_n5`), each as a
  source `.qasm` file and a `_transpiled.qasm` file — 6 members total. Bronze
  is only read and never changed, and the SHA-256 of the archive is
  recorded.
- **Output** (verified against the real course archive, run 2026-09-28):
  - `silver/qasmbench/circuit.parquet`: 6 rows
  - `silver/qasmbench/stabilizer_check.parquet`: 4 rows
  - `silver/qasmbench/conditional_correction.parquet`: 6 rows
  - qasmbench rows in `source_trace.parquet`, `data_issues.parquet` (0 rows
    on the supplied archive), `row_counts.json`, and `run.json`

## Main decisions

**1. One row per circuit variant.** Each `.qasm` member (source or
transpiled) is one row in `circuit.parquet`. `circuit_id` is
`"<benchmark>:<variant>"` (e.g. `qec_sm_n5:source`) — already unique,
readable, and stable, so no extra key is needed. `source_record_id` is
`"qasmbench:<member path>"`.

**2. A circuit is published whole or not at all.** A member that cannot be
decoded as UTF-8 or parsed as OpenQASM 2 contributes no rows to any of the
three tables — not a partial circuit row, and no stabilizer-check or
correction rows either, since those are derived from a circuit that failed
to parse. Its trace row is still written, so the rejection itself is
traceable to Bronze.

**3. Custom gates are expanded before counting or detecting anything.**
OpenQASM 2 allows a custom gate definition whose body is not itself an
executed operation — only where the gate is *invoked* does it count.
`_expand_custom_gates` decomposes every non-standard gate name (up to 8
passes, to handle a custom gate invoking another custom gate) before
`measurement_count`, `two_qubit_gate_count`, and the parity-check/correction
detectors ever run. Without this, a single `syndrome d1,d2,d3,a1,a2;`
statement in `qec_sm_n5` would be missed entirely instead of counted as the
4 CNOTs its body actually contains.

**4. What counts as an explicit parity check.** A check is recorded when
data qubits act as CNOT controls onto one ancilla qubit that is only ever a
CNOT target (never touched by a single-qubit gate), and that ancilla is then
measured into a classical bit — that classical bit is the syndrome bit. On
the supplied archive this fires only for `qec_sm_n5`: ancilla `a[0]` checks
`q[0]`,`q[1]` into `syn[0]`, and `a[1]` checks `q[1]`,`q[2]` into `syn[1]`.
`error_correctiond3_n5` and `qec_en_n5` produce zero check rows on purpose —
neither has a separate ancilla register, and the qubit collecting the parity
in each also receives single-qubit gates and is measured together with the
data qubits into one shared register, so nothing in the file marks a
classical bit as a syndrome bit. The spec asks for checks "where explicit";
Silver does not guess a role for these qubits. See
`docs/decision-log.md`'s qasmbench section for the full register-by-register
evidence.

**5. What counts as a conditional correction.** Each single-qubit gate
inside an `if(register==value)` block is one `conditional_correction` row;
`if(...) x q;` on a whole register produces one row per qubit in that
register (register-wide statements are expanded, matching decision 3's
"count what actually executes" rule). Gates inside an `if` block are
excluded from `measurement_count`/`two_qubit_gate_count`, since they only
run when the condition holds, not unconditionally like the rest of the
circuit body. None of the 3 benchmarks has a conditional two-qubit gate or
measurement, so this exclusion does not change any current count.

**6. One failure mode: unparseable members are recorded and excluded, never
raised.** Unlike `google_qec`'s "a bad companion file stops the whole
experiment" rule, a `qasmbench` member that fails to decode or parse is
written to `data_issues.parquet` (`rule_id = "qasmbench.parse_failed"`, the
member path as the observed value, the parser's message as the reason) and
simply contributes no rows — the other 5 members still publish normally.
This is because qasmbench members are independent circuit files with no
cross-member alignment requirement (unlike Google's shot-indexed companion
files, where one bad file invalidates the whole experiment).

**7. Stable IDs and full tracing.** `source_record_id` for a circuit is
`"qasmbench:<member>"`; a stabilizer-check row's ID appends
`"::stabilizer::<check_id>"` and a correction row's ID appends
`"::correction::<index>"`, so every derived row's ID is still deterministic
from the source member and traces back to it via `source_trace.parquet`. A
rejected member gets exactly one trace row (for the member itself); an
accepted circuit gets one trace row per derived check/correction row too, so
each one can be traced independently.

**8. Reconciliation and reruns.** `row_counts.json` reconciles at the
circuit level (`read = accepted + rejected`); stabilizer-check and
correction counts come only from accepted circuits, since a rejected
circuit's derived rows never existed in the first place. Silver files are
overwritten on each run, and in the shared `results/part1/` files only this
source's rows are replaced (`merge_results_parquet` filters by
`rule_id`/`source_name`), so a second run gives the same records and IDs,
with no duplicates — verified directly by
`test_rerun_keeps_issue_ids_and_counts_reconcile`.

**9. No joins with other sources.** QASMBench's circuits are kept entirely
separate from `google_qec` and `qec_syndromes` in Silver — the spec is
explicit that QASMBench cannot be joined row-by-row to the other two
sources, since they share no experiment identifier. See
`docs/decision-log.md`'s "Rejected candidate relationship" for the evidence
behind this (QASMBench's 5-qubit repetition-code circuits vs. Google's
17-qubit distance-3 surface-code layout).

## Requirements coverage

| Requirement | How it is met |
| --- | --- |
| Required tables, columns and types | Explicit Arrow schemas (`_CIRCUIT_SCHEMA`, `_STABILIZER_SCHEMA`, `_CORRECTION_SCHEMA`); tested by `test_schemas_match_silver_contract` |
| QASM registers, executed operations, parity checks, measurements, and syndrome-controlled recovery are parsed and checked | Decisions 3, 4, 5 |
| Stable `source_record_id` on every row | Decision 7 |
| Every row traced to Bronze | Decision 7 |
| Invalid records kept in `data_issues.parquet` | Decisions 2 and 6 |
| Read = accepted + rejected; rerun gives the same output, no duplicates | Decision 8 |
| No cross-source joins | Decision 9 |

All 7 tests pass: `cd starter && uv run pytest tests/test_silver_qasmbench.py`
(2 of the 7 run only against the real course archive and are skipped
otherwise; verified locally against the real archive — all 7 pass, 0
issues).

## Limitations

- The parity-check detector is a structural heuristic (CNOT-control pattern
  into a measured, single-role ancilla), not a semantic proof that a qubit
  is "the" ancilla — it matches the 3 supplied benchmarks correctly but is
  tied to this specific circuit-writing style, not a general QASM parity-check
  detector.
- Custom-gate expansion caps at 8 passes (`_MAX_GATE_EXPANSION_PASSES`); a
  circuit with a deeper custom-gate nesting than the supplied archive would
  raise rather than expand indefinitely.
