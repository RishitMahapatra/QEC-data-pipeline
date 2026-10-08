-- Part I analysis question 1 (assignment/brief.md):
-- "How do weighted syndrome frequency and logical-error labels change with
-- physical fault rate?"
--
-- Joins gold.gold_syndrome_observation to gold.gold_syndrome_experiment and sums the
-- `quantity` weight (not row counts) per (fault_rate, label) pair, since a
-- single Silver/Gold row can represent millions of physical observations.

SET search_path TO gold;

SELECT
    e.physical_fault_rate,
    o.logical_error_label,
    SUM(o.quantity) AS weighted_count
FROM gold_syndrome_observation o
JOIN gold_syndrome_experiment e ON e.experiment_id = o.experiment_id
GROUP BY e.physical_fault_rate, o.logical_error_label
ORDER BY e.physical_fault_rate, o.logical_error_label;

-- Result (clean run 2026-10-04, run_id f4f249f9-ef6e-4f63-9b74-dcd61520da35;
-- identical to this query's _result.csv, rewritten by every `make run`):
--
-- physical_fault_rate | logical_error_label | weighted_count
-- 0.00001             | false                | 9997660
-- 0.00001             | true                 | 2340
-- 0.00005             | false                | 9988458
-- 0.00005             | true                 | 11542
-- 0.0001              | false                | 9977090
-- 0.0001              | true                 | 22910
-- 0.0005              | false                | 9885597
-- 0.0005              | true                 | 114403
-- 0.001               | false                | 9772982
-- 0.001               | true                 | 227018
-- 0.005               | false                | 8960500
-- 0.005               | true                 | 1039500
-- 0.01                | false                | 8134726
-- 0.01                | true                 | 1865274
--
-- Interpretation: each fault-rate file sums to exactly 10,000,000 weighted
-- observations, matching the "~10M weighted shots per file" from the
-- assignment brief. As physical_fault_rate increases, the weighted share of
-- logical_error_label = true rises steadily, from 0.02% at 0.00001 to 18.7%
-- at 0.01: roughly in proportion to the fault rate at the lower rates, and
-- more slowly at the highest rates.
