-- Part I analysis question 3 (assignment/brief.md):
-- "How does the repetition-code circuit map data qubits to parity-check
-- ancillas, syndrome bits, and conditional corrections?"
--
-- Joins gold_qasmbench_circuit, gold_qasmbench_stabilizer_check,
-- gold_qasmbench_register, and gold_qasmbench_conditional_correction (4
-- Gold tables) for the one QASMBench benchmark with an explicit
-- parity-check/correction structure, qec_sm_n5 (see docs/decision-log.md's
-- qasmbench section for why the other two benchmarks produce no check/
-- correction rows). The register join resolves each check's syndrome_bit
-- (e.g. "syn[0]") back to its declared register width, confirming the
-- 2-bit syndrome register is exactly wide enough for condition_value's
-- 1..3 range (2^2 - 1).
--
-- All Gold tables live in the dedicated `gold` schema (team decision, see
-- docs/decision-log.md) - set search_path before running this ad hoc, or
-- schema-qualify the table names.

SET search_path TO gold;

SELECT
    c.circuit_id,
    c.variant,
    sc.ancilla_qubit,
    sc.data_qubits,
    sc.syndrome_bit,
    r.register_size AS syndrome_register_size,
    cc.condition_value,
    cc.gate,
    cc.target_qubit
FROM gold_qasmbench_circuit c
JOIN gold_qasmbench_stabilizer_check sc ON sc.circuit_id = c.circuit_id
JOIN gold_qasmbench_register r
    ON r.circuit_id = c.circuit_id
   AND r.register_kind = 'creg'
   AND r.register_name = split_part(sc.syndrome_bit, '[', 1)
JOIN gold_qasmbench_conditional_correction cc ON cc.circuit_id = c.circuit_id
WHERE c.benchmark_name = 'qec_sm_n5'
ORDER BY c.variant, sc.ancilla_qubit, cc.condition_value;

-- Result (clean run 2026-10-04, run_id f4f249f9-ef6e-4f63-9b74-dcd61520da35;
-- identical to this query's _result.csv, rewritten by every `make run`):
--
-- circuit_id            | variant    | ancilla_qubit | data_qubits | syndrome_bit | syndrome_register_size | condition_value | gate | target_qubit
-- qec_sm_n5:source      | source     | a[0]          | {q[0],q[1]} | syn[0]       | 2                      | 1               | x    | q[0]
-- qec_sm_n5:source      | source     | a[0]          | {q[0],q[1]} | syn[0]       | 2                      | 2               | x    | q[2]
-- qec_sm_n5:source      | source     | a[0]          | {q[0],q[1]} | syn[0]       | 2                      | 3               | x    | q[1]
-- qec_sm_n5:source      | source     | a[1]          | {q[1],q[2]} | syn[1]       | 2                      | 1               | x    | q[0]
-- qec_sm_n5:source      | source     | a[1]          | {q[1],q[2]} | syn[1]       | 2                      | 2               | x    | q[2]
-- qec_sm_n5:source      | source     | a[1]          | {q[1],q[2]} | syn[1]       | 2                      | 3               | x    | q[1]
-- qec_sm_n5:transpiled  | transpiled | a[0]          | {q[0],q[1]} | syn[0]       | 2                      | 1               | x    | q[0]
-- qec_sm_n5:transpiled  | transpiled | a[0]          | {q[0],q[1]} | syn[0]       | 2                      | 2               | x    | q[2]
-- qec_sm_n5:transpiled  | transpiled | a[0]          | {q[0],q[1]} | syn[0]       | 2                      | 3               | x    | q[1]
-- qec_sm_n5:transpiled  | transpiled | a[1]          | {q[1],q[2]} | syn[1]       | 2                      | 1               | x    | q[0]
-- qec_sm_n5:transpiled  | transpiled | a[1]          | {q[1],q[2]} | syn[1]       | 2                      | 2               | x    | q[2]
-- qec_sm_n5:transpiled  | transpiled | a[1]          | {q[1],q[2]} | syn[1]       | 2                      | 3               | x    | q[1]
-- (12 rows)
--
-- Interpretation: ancilla a[0] collects the parity of data qubits q[0] and
-- q[1] into syn[0]; ancilla a[1] collects the parity of q[1] and q[2] into
-- syn[1] - q[1] participates in both checks, which is what makes this a
-- repetition code rather than two independent checks. Every syndrome value
-- from 1 to 3 (the full non-zero range of the 2-bit syn register) has a
-- matching conditional correction: value 1 (syn[0] alone) corrects q[0],
-- value 2 (syn[1] alone) corrects q[2], and value 3 (both bits set)
-- corrects q[1] - the qubit shared by both checks. This is exactly the
-- repetition-code decoding rule (a lone flipped check points at the qubit
-- next to it; both checks flipping points at the shared qubit), read
-- directly off the circuit rather than assumed. Source and transpiled
-- variants agree row-for-row, confirming transpilation did not change the
-- circuit's parity-check/correction structure. This is a structural
-- association read from the circuit definition, not a claim about any
-- run's actual measured syndromes.
