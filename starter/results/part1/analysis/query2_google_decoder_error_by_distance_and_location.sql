-- Part I analysis question 2 (assignment/brief.md):
-- "How do the supplied decoder logical-error rates compare by code distance
-- and distance-three processor location?"
--
-- A decoder makes a logical error on a shot when its predicted logical flip
-- differs from the actual one. Gold already stores that per shot and per
-- decoder as gold_google_decoder_prediction.was_wrong, so the logical-error
-- rate is simply (shots where was_wrong) / (all shots), grouped by
-- experiment and decoder. Each experiment is one (distance, processor
-- location) combination: four distance-3 locations and one distance-5
-- location.
--
-- Joins three Gold tables:
--   gold_google_experiment          distance and processor location (center_row, center_col)
--   gold_google_decoder_prediction  one row per shot per decoder, with was_wrong
--   gold_google_shot                actual_observable_flip, for the majority baseline
--
-- Output columns:
--   logical_error_rate            this decoder at this location
--   rank_at_location              1 = fewest errors among the four decoders at this location
--   pooled_rate_for_distance      this decoder over all shots of that distance (for
--                                 distance 3: the four locations together, 200,000 shots)
--   majority_baseline_error_rate  error rate of always guessing the more common actual
--                                 outcome at this location: the reference a decoder must beat
--
-- Scope: all 50,000 shots of every experiment. This is a description of the
-- supplied predictions, not a model evaluation; Part II compares decoders on
-- the held-out test split only.

SET search_path TO gold;

-- Step 1: count shots and decoder errors per (experiment, decoder).
-- COUNT(*) FILTER where conts only the rows matching the condition.
WITH decoder_errors AS (
    SELECT
        e.experiment_id,
        e.distance,
        e.center_row,
        e.center_col,
        p.decoder_name,
        COUNT(*)                            AS shots,
        COUNT(*) FILTER (WHERE p.was_wrong) AS logical_errors
    FROM gold_google_experiment e
    JOIN gold_google_decoder_prediction p ON p.experiment_id = e.experiment_id
    GROUP BY e.experiment_id, e.distance, e.center_row, e.center_col, p.decoder_name
),

-- Step 2: majority baseline per experiment. AVG of the 0/1 actual flip is the
-- share of shots that flipped; always guessing the more common outcome is
-- wrong on the smaller share, hence LEAST(share, 1 - share).
majority_baseline AS (
    SELECT
        experiment_id,
        LEAST(AVG(actual_observable_flip::INT), 1 - AVG(actual_observable_flip::INT)) AS baseline_error_rate
    FROM gold_google_shot
    GROUP BY experiment_id
)

-- Step 3: turn counts into rates and compare. Window functions (OVER ...)
-- add a value computed over a group of rows without collapsing them, so each
-- row keeps its per-location rate and also gets its rank at that location
-- and its pooled rate for the whole distance.
SELECT
    d.distance,
    d.center_row,
    d.center_col,
    d.decoder_name,
    d.shots,
    d.logical_errors,
    ROUND(d.logical_errors::NUMERIC / d.shots, 4)                         AS logical_error_rate,
    RANK() OVER (PARTITION BY d.experiment_id ORDER BY d.logical_errors)  AS rank_at_location,
    ROUND(SUM(d.logical_errors) OVER per_distance::NUMERIC
          / SUM(d.shots) OVER per_distance, 4)                            AS pooled_rate_for_distance,
    ROUND(b.baseline_error_rate, 4)                                       AS majority_baseline_error_rate
FROM decoder_errors d
JOIN majority_baseline b ON b.experiment_id = d.experiment_id
WINDOW per_distance AS (PARTITION BY d.distance, d.decoder_name)
ORDER BY d.distance, d.center_row, d.center_col, rank_at_location, d.decoder_name;

-- Result (clean run 2026-10-04, run_id f4f249f9-ef6e-4f63-9b74-dcd61520da35;
-- identical to this query's _result.csv, rewritten by every `make run`):
--
-- distance | center_row | center_col | decoder_name               | shots | logical_errors | logical_error_rate | rank_at_location | pooled_rate_for_distance | majority_baseline_error_rate
-- 3        | 3          | 5          | tensor_network_contraction | 50000 | 20114          | 0.4023             | 1                | 0.3970                   | 0.4903
-- 3        | 3          | 5          | belief_matching            | 50000 | 20125          | 0.4025             | 2                | 0.3997                   | 0.4903
-- 3        | 3          | 5          | correlated_matching        | 50000 | 20971          | 0.4194             | 3                | 0.4190                   | 0.4903
-- 3        | 3          | 5          | pymatching                 | 50000 | 21708          | 0.4342             | 4                | 0.4345                   | 0.4903
-- 3        | 5          | 3          | tensor_network_contraction | 50000 | 20652          | 0.4130             | 1                | 0.3970                   | 0.4958
-- 3        | 5          | 3          | belief_matching            | 50000 | 20796          | 0.4159             | 2                | 0.3997                   | 0.4958
-- 3        | 5          | 3          | correlated_matching        | 50000 | 21259          | 0.4252             | 3                | 0.4190                   | 0.4958
-- 3        | 5          | 3          | pymatching                 | 50000 | 22159          | 0.4432             | 4                | 0.4345                   | 0.4958
-- 3        | 5          | 7          | tensor_network_contraction | 50000 | 19196          | 0.3839             | 1                | 0.3970                   | 0.4946
-- 3        | 5          | 7          | belief_matching            | 50000 | 19598          | 0.3920             | 2                | 0.3997                   | 0.4946
-- 3        | 5          | 7          | correlated_matching        | 50000 | 20912          | 0.4182             | 3                | 0.4190                   | 0.4946
-- 3        | 5          | 7          | pymatching                 | 50000 | 21532          | 0.4306             | 4                | 0.4345                   | 0.4946
-- 3        | 7          | 5          | belief_matching            | 50000 | 19418          | 0.3884             | 1                | 0.3997                   | 0.4915
-- 3        | 7          | 5          | tensor_network_contraction | 50000 | 19436          | 0.3887             | 2                | 0.3970                   | 0.4915
-- 3        | 7          | 5          | correlated_matching        | 50000 | 20648          | 0.4130             | 3                | 0.4190                   | 0.4915
-- 3        | 7          | 5          | pymatching                 | 50000 | 21497          | 0.4299             | 4                | 0.4345                   | 0.4915
-- 5        | 5          | 5          | tensor_network_contraction | 50000 | 19775          | 0.3955             | 1                | 0.3955                   | 0.5000
-- 5        | 5          | 5          | belief_matching            | 50000 | 20074          | 0.4015             | 2                | 0.4015                   | 0.5000
-- 5        | 5          | 5          | correlated_matching        | 50000 | 21283          | 0.4257             | 3                | 0.4257                   | 0.5000
-- 5        | 5          | 5          | pymatching                 | 50000 | 22551          | 0.4510             | 4                | 0.4510                   | 0.5000
--
-- Interpretation frtom the resuts:

--  Every decoder beats the majority baseline everywhere (about 0.38-0.45 vs 0.49-0.50), but all rates are far from zero. These are 25-round experiments: errors keep accumulating round after round, and a rate of 0.5 would mean the logical outcome is a coin flip.

--  Decoder order is almost the same at every location: tensor-network contraction and belief matching are best and within 0.01 of each other (belief matching is marginally ahead at (7,5)), then correlated matching, then PyMatching. The gap between best and worst is about 0.03-0.06.

--  Distance-3 location matters: for the same decoder, rates differ by up to
--   about 0.03 between locations; (5,7) and (7,5) are lowest, (5,3) highest. The data does not say why; the locations are different physical qubits on the same processor.

--  Distance 5 is not clearly better than distance 3 in this release: its rates sit inside the range of the four distance-3 locations (best decoder 0.3955 vs pooled distance-3 0.3970; PyMatching is worse at distance 5). With one distance-5 experiment at one location, distance and location cannot be separated here, so no conclusion about distance itself is drawn. These are associations in the supplied data, not causal claims.
