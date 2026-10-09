'''
This code file is for the integration of relational database 

According to the rubrics, it includes:
1) creation of the user defined tables
2) primary and foreign keys
3) includes value checks and indexes

Note: Repeated loads must not create duplicate records

Sources loaded here, each into its own tables inside the shared `gold` schema:
- google_qec     -> gold_google_experiment, gold_google_shot, gold_google_decoder, gold_google_decoder_prediction
- qec_syndromes  -> gold_syndrome_experiment, gold_syndrome_pattern, gold_syndrome_observation
- qasmbench      -> gold_qasmbench_circuit, gold_qasmbench_register, gold_qasmbench_stabilizer_check, gold_qasmbench_conditional_correction
run() loads every source in ONE transaction, so the whole Gold update is all-or-nothing.
After the loads, run() creates v_qec_experiment_catalog, a UNION view that lists every source's
experiments/circuits side by side in the shared QEC vocabulary (code_family, distance, rounds).
'''


import hashlib # SHA-256 for Silver input hashes and syndrome observation ids
import io #used to read the raw MinIO bytes so pyarrow can read them as a file
import json # qasmbench's register_declarations Silver column is JSON text, decoded here

import pyarrow as pa  # in-memory tabular data type used by Silver
import pyarrow.parquet as pq  # reads Parquet files into pyarrow tables

from quantum_lake_student import connections # for postgres_connection() helper
from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client #for creation of minio client
from quantum_lake_student.models import RESULTS_PART1, StageResult, merge_json_file, code_revision
from quantum_lake_student.stages.prepare_data import SILVER_SYNDROMES_PATH

'''
For the google_qec gold stage -> takes the two validated silver parquet tables. Since Parquet tables are for indexing and storage, running queries is not handled. For that reason, we implement PostgreSQL at the gold stage. Takes the two validated Silver Parquet tables
(experiment.parquet, shot.parquet) and loads them into real PostgreSQL tables that enforce keys, foreign keys, and CHECK constraints.

Load is all-or-nothing (rollback on any failure), and idempotent (rerun replaces the rows, doesn't duplicate).
'''

# ---------------------------------------------------------------------------
"""
Shared QEC vocabulary

 The three sources share QEC concepts but no row-level identifier (assignment/faq.md: "Use clear shared names without inventing a match").
 So Gold conforms names instead of joining rows: every source's experiment/circuit table carries the same two columns, with the same name,
 type and allowed values:
   code_family  'surface_code' or 'repetition_code'
   distance     code distance (positive integer)
 They are labels, never join keys: no foreign key connects two sources.
 Full list of conformed terms: reports/gold_stage/gold_shared_vocabulary.md
"""
# ---------------------------------------------------------------------------

GOOGLE_CODE_FAMILY = "surface_code"

# DECODER_NAMES: the four decoders supplied for google_qec. Pulled out as a module-level constant so both the schema design and the load logic agree on the same four names
DECODER_NAMES = (
    "belief_matching",
    "correlated_matching",
    "pymatching",
    "tensor_network_contraction",
)

# The DDL that creates both Gold tables: gold_google_experiment and gold_google_shot. Using IF NOT EXISTS to carry out no operations if the table already exists -> Safer choice 
#gold_google_experiment has one row for one experiment, whereas gold_google_shot has many rows for a single experiment
GOOGLE_GOLD_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gold_google_experiment ( --for each experiment directory
    experiment_id       TEXT PRIMARY KEY, -- one row per experiment
    source_record_id TEXT NOT NULL UNIQUE, -- traces which Silver row this Gold row came from
    basis                TEXT NOT NULL CHECK (basis in('X','Z')), -- logical measurement basis, only X or Z
    code_family          TEXT NOT NULL CHECK (code_family IN ('surface_code', 'repetition_code')), -- shared vocabulary label, same column in every source
    distance             INTEGER NOT NULL CHECK (distance > 0), --the distance must be positive 
    rounds                INTEGER NOT NULL CHECK (rounds > 0), -- QEC rounds must be positive
    shots                 INTEGER NOT NULL CHECK (shots > 0), --shot count must be positive
    center_row            INTEGER NOT NULL, -- processor location, row coordinate
    center_col            INTEGER NOT NULL, --processor location, column coordinate
    measurement_count     INTEGER NOT NULL CHECK (measurement_count > 0), --bits per shot in measurements.b8
        detector_count        INTEGER NOT NULL CHECK (detector_count > 0), --bits per shot in detection_events.b8 
    CHECK (
        (distance = 3 AND detector_count = 200) OR (distance = 5 AND detector_count = 600)
    ), -- This checks if the actual counts match the properties.yml where the fact is: For distance = 3, there will be 200 detectors and for distance = 5, there will be 600 detectors. PostgreSQL will automatically reject any files with mismatched distance and detector count.
    
      UNIQUE(basis,distance, rounds, center_row, center_col), -- this ensures that no two experiments can share the same basis/distance/rounds/location
    UNIQUE (experiment_id, measurement_count, detector_count) -- required so gold_google_shot's composite foreign key can reference this exact triple
);

COMMENT ON TABLE gold_google_experiment IS
    'One row represents one Google hardware experiment configuration (a distance/rounds/processor-location combination).';

CREATE TABLE IF NOT EXISTS gold_google_shot ( -- for the single shots
    experiment_id                          TEXT NOT NULL REFERENCES gold_google_experiment(experiment_id), -- which experiment this shot was run in
    measurement_count INTEGER NOT NULL,  -- copied from the experiment row so byte-length checks below can reference it without a cross-table query
    detector_count INTEGER NOT NULL, -- same reasoning as the above -> measurement count 
    source_record_id TEXT NOT NULL UNIQUE, -- traces which Silver row this Gold row came from
    shot_index                             INTEGER NOT NULL CHECK (shot_index >= 0), -- can be zero based and non negative
    
    sweep_bits                             BYTEA NOT NULL, -- configuration bits for this shot
        detector_bits                          BYTEA NOT NULL CHECK (octet_length(detector_bits) = CEIL(detector_count / 8.0)), -- must match THIS shot's own detector_count exactly, not just any valid length
    
    measurement_bits                       BYTEA NOT NULL CHECK (octet_length(measurement_bits) = CEIL(measurement_count / 8.0)), -- same idea for measurement bits
    
        detector_event_count                   INTEGER NOT NULL CHECK (detector_event_count >= 0 AND detector_event_count <= detector_count), --counts the set of detector bits; can never exceed the total number of detectors
    actual_observable_flip                 BOOLEAN NOT NULL, --ground truth outcome 
    PRIMARY KEY (experiment_id, shot_index),  -- composite: a shot is only unique WITHIN its experiment

    

    --adding the foreign key: to enforce that these copies actually match the real experiment row
    FOREIGN KEY(experiment_id, measurement_count, detector_count)
        REFERENCES gold_google_experiment (experiment_id, measurement_count, detector_count)
        -- if a shot's copied counts don't match the actual experiment row, the insert is rejected

      
);
-- NOTE: the four decoder prediction columns from Silver live in gold_google_decoder_prediction below, so that "what a decoder guessed" and "whether it was wrong" are their own table instead of four repeated yes/no columns on shot

COMMENT ON TABLE gold_google_shot IS
    'One row represents one hardware shot (one circuit execution) within one Google experiment, with its raw bits and actual outcome. Decoder predictions live in gold_google_decoder_prediction.';

CREATE INDEX IF NOT EXISTS idx_google_shot_experiment_id ON gold_google_shot(experiment_id);

-- One row per decoder algorithm (4 rows total for google_qec)

CREATE TABLE IF NOT EXISTS gold_google_decoder (
    decoder_name TEXT PRIMARY KEY -- name of the supplied decoder, taken from its obs_flips_predicted_by_<decoder>.01 file
);

COMMENT ON TABLE gold_google_decoder IS
    'One row per decoder algorithm available for google_qec (belief_matching, correlated_matching, pymatching, tensor_network_contraction).';

-- One row per shot per decoder, separating "what it guessed" from "whether it was wrong"

CREATE TABLE IF NOT EXISTS gold_google_decoder_prediction (
    experiment_id     TEXT NOT NULL, -- experiment of the shot being decoded
    shot_index        INTEGER NOT NULL, -- zero-based shot position within that experiment; with experiment_id identifies the shot
    decoder_name      TEXT NOT NULL REFERENCES gold_google_decoder(decoder_name), -- which of the 4 supplied decoders made this prediction
    predicted_flip    BOOLEAN NOT NULL, -- the decoder's guess: TRUE if it predicts the logical observable flipped
    was_wrong         BOOLEAN NOT NULL, -- decoder mistake: TRUE when predicted_flip differs from the shot's actual_observable_flip
    PRIMARY KEY (experiment_id, shot_index, decoder_name), -- one prediction per decoder per shot
    FOREIGN KEY (experiment_id, shot_index) REFERENCES gold_google_shot(experiment_id, shot_index) -- every prediction belongs to a loaded shot
);
COMMENT ON TABLE gold_google_decoder_prediction IS
    'One row per shot per decoder: what that decoder predicted, and whether it disagreed with the actual outcome.';

CREATE INDEX IF NOT EXISTS idx_decoder_prediction_decoder ON gold_google_decoder_prediction(decoder_name);
"""

def _read_silver_parquet(settings, object_name: str) -> pa.Table: #this is a helper function
    """Read one Silver Parquet table from the lake (MinIO bucket or local lake folder)."""

    # Defines a helper function. Takes 2 inputs (settings = env config, object_name = which file to read) and returns a pyarrow Table (a tabular data structure like a DataFrame).
    
    #There are two files maintained, local-disk lake (test/dev) and MinIO (real environment).
    if settings.lake_backend == "local":
        #opening the local backend file as parquet table only 
        return pq.read_table(settings.local_lake_root / object_name)
    # MinIO backend: pull the object's bytes over HTTP, then parse them in memory.
    client = minio_client(settings)
    resp = client.get_object(settings.s3_bucket, object_name) #starts streaming
    try:
        data = resp.read() #reading the entire object to the memory
    finally:
        resp.close() 
        resp.release_conn()
    return pq.read_table(io.BytesIO(data))  # wrap bytes as a file-like object for pyarrow

#hash the Parquet file's raw bytes when reading.
def _read_silver_parquet_with_hash (settings, object_name: str) -> tuple[pa.Table, str]:
    """Same as _read_silver_parquet, but also returns the SHA-256 of the raw file bytes."""
    #trying to check the source and accodingly assign the data
    if settings.lake_backend=="local":
        data = (settings.local_lake_root / object_name).read_bytes() #reads the bytes directly from the local files
    else:
        client = minio_client(settings) #sets up a new client based on the settings 
        resp = client.get_object(settings.s3_bucket, object_name) #initialises a bucket 
        try:
            data = resp.read() #reads the data from the bucket
        finally: #for closing and releasing the connection
            resp.close()
            resp.release_conn()
    return pq.read_table(io.BytesIO(data)), hashlib.sha256(data).hexdigest() #this returns the SHA 256 hashed version of the parquet file's raw bytes when reading
    
def load_google_gold(conn, settings) -> tuple[int, int, int, str, str]: #this function loads the silver data to gold
    # Args:
    # conn: takes the PostgreSQL connection, settings = env config  returns (experiment count, shot count, decoder prediction count, experiment file hash, shot file hash)
    
    """Load Silver's google_qec Parquet tables into Postgres, all-or-nothing.

    Truncates both tables first, then inserts fresh - so a rerun never
    duplicates rows, and if anything fails partway, the transaction rolls
    back and the previous complete data stays in place.
    """
    experiment_table, experiment_hash = _read_silver_parquet_with_hash(settings, "silver/google_qec/experiment.parquet")
    shot_table, shot_hash = _read_silver_parquet_with_hash(settings, "silver/google_qec/shot.parquet")

    with conn.cursor() as cur: # Open a database cursor to run SQL commands. TRUNCATE wipes all four Gold tables - CASCADE handles the dependency order automatically
        cur.execute(
            "TRUNCATE TABLE gold_google_decoder_prediction, gold_google_shot, "
            "gold_google_experiment, gold_google_decoder RESTART IDENTITY CASCADE;"
        ) #using TRUNCATE as its faster than DELETE

        cur.executemany( ## executemany runs the same SQL statement many times with different values

            
            """
            INSERT INTO gold_google_experiment
                (experiment_id, source_record_id, basis, code_family, distance, rounds, shots, center_row, center_col, measurement_count, detector_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            # contains the template for insertion of values
            [
                (r["experiment_id"], r["source_record_id"], r["basis"], GOOGLE_CODE_FAMILY, r["distance"], r["rounds"], r["shots"],
                 r["center_row"], r["center_col"], r["measurement_count"], r["detector_count"])
                for r in experiment_table.to_pylist()
            ], # Build a list of tuples, one per experiment row.  Each tuple's values match the %s placeholders in order.

             # experiment_table.to_pylist() converts the pyarrow table into a list of Python dicts (one dict per row) that we can pull values from by column name.
        )

                # build an experiment_id -> (measurement_count, detector_count) lookup so each
        # shot row can carry its own experiment's counts, needed for the composite FK check
        experiment_counts = {
            r["experiment_id"]: (r["measurement_count"], r["detector_count"])
            for r in experiment_table.to_pylist()
        }

        cur.executemany(
            """
            INSERT INTO gold_google_shot
                (experiment_id, source_record_id, shot_index, measurement_count, detector_count,
                 measurement_bits, sweep_bits, detector_bits, detector_event_count, actual_observable_flip)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    r["experiment_id"], r["source_record_id"], r["shot_index"],
                    *experiment_counts[r["experiment_id"]],
                    r["measurement_bits"], r["sweep_bits"], r["detector_bits"],
                    r["detector_event_count"], r["actual_observable_flip"],
                )
                for r in shot_table.to_pylist()
            ],
        )

        # Load the decoder table and the decoder_prediction table

        # 4 rows, one per decoder name
        cur.executemany(
            "INSERT INTO gold_google_decoder (decoder_name) VALUES (%s)",
            [(name,) for name in DECODER_NAMES],
        )

        # one row per shot per decoder: what it predicted, and whether that disagreed with the actual outcome. They are kept as two separate facts instead of mixing "guess" and "correctness" into one column
        decoder_prediction_rows = [
            (r["experiment_id"], r["shot_index"], name, r[f"{name}_prediction"],
             r[f"{name}_prediction"] != r["actual_observable_flip"])
            for r in shot_table.to_pylist()
            for name in DECODER_NAMES
        ]

        cur.executemany(
            """
            INSERT INTO gold_google_decoder_prediction
                (experiment_id, shot_index, decoder_name, predicted_flip, was_wrong)
            VALUES (%s, %s, %s, %s, %s)
            """,
            decoder_prediction_rows,
        )

    return experiment_table.num_rows, shot_table.num_rows, len(decoder_prediction_rows), experiment_hash, shot_hash


# ---------------------------------------------------------------------------
# qec_syndromes Gold tables
# ---------------------------------------------------------------------------

SYNDROMES_SOURCE_NAME = "qec_syndromes"

# The source documents these files as simulated distance-3 surface-code syndromes (assignment/data-sources.md); the filenames only encode distance,
# so the family is a source-level constant (see "Shared QEC vocabulary" above).
SYNDROMES_CODE_FAMILY = "surface_code"

# gold_syndrome_experiment table with one row per fault-rate file, with checks that the rate is a probability (between 0 and 1) and rounds and checks both equal 4.
# code_family/distance are the shared-vocabulary columns every source carries.
SYNDROME_EXPERIMENT_DDL = """
CREATE TABLE IF NOT EXISTS gold_syndrome_experiment (
    experiment_id       TEXT PRIMARY KEY, -- "qec_syndromes:<csv filename>", one per fault-rate file, stable across runs
    code_family         TEXT NOT NULL CHECK (code_family IN ('surface_code', 'repetition_code')), -- shared vocabulary label, same column in every source
    distance            INT NOT NULL CHECK (distance > 0), -- code distance from the filename (d-3); shared vocabulary
    physical_fault_rate NUMERIC NOT NULL CHECK (physical_fault_rate > 0 AND physical_fault_rate < 1), -- simulated per-operation fault probability from the filename (pfr-...)
    round_count         INT NOT NULL CHECK (round_count = 4), -- syndrome-measurement rounds per sample; always 4 in this source
    check_count         INT NOT NULL CHECK (check_count = 4) -- stabilizer checks measured per round; always 4 in this source
);
COMMENT ON TABLE gold_syndrome_experiment IS
    'One row represents one simulated physical-fault-rate experiment (one source CSV file).';
"""

# gold_syndrome_pattern: one row per distinct 16-bit pattern, plus how many of its 16 checks actually fired. The CHECK is a DB-level guarantee that a pattern is
# really 16 bytes and each byte is 0 or 1 (hex "00" or "01"), on top of what

SYNDROME_PATTERN_DDL = """
CREATE TABLE IF NOT EXISTS gold_syndrome_pattern (
    syndrome_bits          BYTEA PRIMARY KEY CHECK (encode(syndrome_bits, 'hex') ~ '^(0[01]){16}$'), -- 16 one-byte 0/1 values, round first then check (4 rounds x 4 checks)
    triggered_check_count  INT NOT NULL CHECK (triggered_check_count BETWEEN 0 AND 16) -- how many of the 16 check results are 1
);
COMMENT ON TABLE gold_syndrome_pattern IS
    'One row represents one distinct 16-bit syndrome pattern, seen in any experiment.';
"""


SYNDROME_OBSERVATION_DDL = """
CREATE TABLE IF NOT EXISTS gold_syndrome_observation (
    observation_id       TEXT PRIMARY KEY, -- deterministic hash of (experiment_id, syndrome_bits, label), stable across runs
    experiment_id        TEXT NOT NULL REFERENCES gold_syndrome_experiment (experiment_id), -- which fault-rate file this row came from
    syndrome_bits         BYTEA NOT NULL REFERENCES gold_syndrome_pattern (syndrome_bits), -- the 16-bit syndrome pattern observed
    logical_error_label   BOOLEAN NOT NULL, -- source `labels` value: TRUE if a logical error occurred
    quantity              BIGINT NOT NULL CHECK (quantity > 0), -- physical observations this row represents; the weight for every aggregate
    source_record_id      TEXT NOT NULL UNIQUE, -- traces which Silver row this Gold row came from
    UNIQUE (experiment_id, syndrome_bits, logical_error_label) -- same pattern may appear with both labels, but only once per label per experiment
);
COMMENT ON TABLE gold_syndrome_observation IS
    'One row represents one (experiment, syndrome pattern, logical-error label) combination and its quantity weight.';
CREATE INDEX IF NOT EXISTS idx_gold_syndrome_observation_experiment ON gold_syndrome_observation (experiment_id);
CREATE INDEX IF NOT EXISTS idx_gold_syndrome_observation_syndrome_bits ON gold_syndrome_observation (syndrome_bits);
"""


def _triggered_check_count(syndrome_bits: bytes) -> int:
    # returns how many of the 16 bits are set to 1
    return sum(syndrome_bits)


def _syndrome_observation_id(experiment_id: str, syndrome_bits: bytes, label: bool) -> str:
    # hashes experiments + pattern + label into one deterministic id
    # that is easier to reference later instead of a composite key
    key = f"{experiment_id}:{syndrome_bits.hex()}:{label}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def create_syndrome_tables(cursor) -> None:
    # sends the three "create table" to postgress
    cursor.execute(SYNDROME_EXPERIMENT_DDL)
    cursor.execute(SYNDROME_PATTERN_DDL)
    cursor.execute(SYNDROME_OBSERVATION_DDL)


def _check_no_silent_collisions(rows: list[dict]) -> None:
    # raise an error if two silver rows hash to the same observation_id 
    # but have different source_record_id
    seen: dict[str, str] = {}
    for r in rows:
        observation_id = _syndrome_observation_id(
            r["experiment_id"], r["syndrome_bits"], r["logical_error_label"]
        )
        existing_source = seen.get(observation_id)
        if existing_source is not None and existing_source != r["source_record_id"]:
            raise ValueError(
                f"Silver rows {existing_source!r} and {r['source_record_id']!r} both map to "
                f"observation_id {observation_id!r}"
            )
        seen[observation_id] = r["source_record_id"]


def load_syndrome_tables(cursor, silver_table) -> None:
    # loads data from silver(parquet) into gold(postgres)
    rows = silver_table.to_pylist()
    _check_no_silent_collisions(rows)

    experiments = {
        r["experiment_id"]: (r["distance"], r["physical_fault_rate"], r["round_count"], r["check_count"])
        for r in rows
    }
    for experiment_id, (distance, rate, rounds, checks) in experiments.items():
        cursor.execute(
            """
            INSERT INTO gold_syndrome_experiment
                (experiment_id, code_family, distance, physical_fault_rate, round_count, check_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id) DO UPDATE
            SET code_family = EXCLUDED.code_family,
                distance = EXCLUDED.distance,
                physical_fault_rate = EXCLUDED.physical_fault_rate
            """,
            (experiment_id, SYNDROMES_CODE_FAMILY, distance, rate, rounds, checks),
        )

    for bits in {r["syndrome_bits"] for r in rows}:
        cursor.execute(
            """
            INSERT INTO gold_syndrome_pattern (syndrome_bits, triggered_check_count)
            VALUES (%s, %s)
            ON CONFLICT (syndrome_bits) DO NOTHING
            """,
            (bits, _triggered_check_count(bits)),
        )

    for r in rows:
        observation_id = _syndrome_observation_id(
            r["experiment_id"], r["syndrome_bits"], r["logical_error_label"]
        )
        cursor.execute(
            """
            INSERT INTO gold_syndrome_observation
                (observation_id, experiment_id, syndrome_bits, logical_error_label, quantity, source_record_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (observation_id) DO UPDATE
            SET quantity = EXCLUDED.quantity, source_record_id = EXCLUDED.source_record_id
            """,
            (observation_id, r["experiment_id"], r["syndrome_bits"], r["logical_error_label"], r["quantity"], r["source_record_id"]),
        )

    # rows that used to be in Silver but no longer shouldn't linger in Gold
    current_source_ids = [r["source_record_id"] for r in rows]
    cursor.execute(
        "DELETE FROM gold_syndrome_observation WHERE NOT (source_record_id = ANY(%s))",
        (current_source_ids,),
    )
    current_experiment_ids = list(experiments)
    cursor.execute(
        "DELETE FROM gold_syndrome_experiment WHERE NOT (experiment_id = ANY(%s))",
        (current_experiment_ids,),
    )
    current_patterns = list({r["syndrome_bits"] for r in rows})
    cursor.execute(
        "DELETE FROM gold_syndrome_pattern WHERE NOT (syndrome_bits = ANY(%s))",
        (current_patterns,),
    )


# ---------------------------------------------------------------------------
# qasmbench Gold tables
# ---------------------------------------------------------------------------

QASMBENCH_SOURCE_NAME = "qasmbench"
"""
Silver stores register_declarations as a JSON blob; Gold splits it into one typed row per register so it can be queried and constrained properly.

data_qubits is kept as a TEXT[] column rather than a junction table -
there are only ~8 (qubit, check) pairs in the whole archive, so a separate table would add overhead for no real gain. We use cardinality() in the
CHECK (not array_length) because array_length returns NULL for empty arrays, which would silently pass the constraint.

Only circuits with an explicit parity-check structure get a code_family/distance label.
 qec_sm_n5 is a distance-3 repetition code (3 data qubits, 2 parity checks). The other two benchmarks have no explicit structure, so they stay NULL.
"""
QASMBENCH_CODE_LABELS = {
    "qec_sm_n5": ("repetition_code", 3),
}

QASMBENCH_GOLD_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gold_qasmbench_circuit ( -- one row per parsed QASMBench circuit variant
    circuit_id            TEXT PRIMARY KEY, -- "<benchmark>:<variant>", already unique and stable
    source_record_id      TEXT NOT NULL UNIQUE, -- traces which Silver row this Gold row came from
    benchmark_name         TEXT NOT NULL, -- which of the 3 QASMBench circuits this is
    variant                TEXT NOT NULL CHECK (variant IN ('source', 'transpiled')), -- as published, or after transpilation
    qubit_count            INTEGER NOT NULL CHECK (qubit_count > 0), -- declared qubits must be positive
    measurement_count      INTEGER NOT NULL CHECK (measurement_count >= 0), -- executed measure statements, expanded
    two_qubit_gate_count   INTEGER NOT NULL CHECK (two_qubit_gate_count >= 0), -- executed 2-qubit gates, expanded
    code_family            TEXT CHECK (code_family IN ('surface_code', 'repetition_code')), -- shared vocabulary; NULL when the file does not make the code explicit
    distance               INTEGER CHECK (distance > 0), -- shared vocabulary; NULL for the same reason

    CHECK ((code_family IS NULL) = (distance IS NULL)), -- a circuit is either labelled with both or with neither
    UNIQUE (benchmark_name, variant) -- a second uniqueness rule on the same fact as circuit_id, enforced independently
);

COMMENT ON TABLE gold_qasmbench_circuit IS
    'One row represents one parsed QASMBench circuit variant (a benchmark, as published or after transpilation).';

CREATE TABLE IF NOT EXISTS gold_qasmbench_register ( -- one row per declared qreg/creg of one circuit variant
    circuit_id      TEXT NOT NULL REFERENCES gold_qasmbench_circuit(circuit_id) ON DELETE CASCADE, -- which circuit variant declares this register
    register_kind   TEXT NOT NULL CHECK (register_kind IN ('qreg', 'creg')), -- quantum or classical register
    register_name   TEXT NOT NULL, -- e.g. "q", "a", "c", "syn"
    register_size   INTEGER NOT NULL CHECK (register_size > 0), -- declared register width must be positive

    PRIMARY KEY (circuit_id, register_kind, register_name) -- a register name is only unique within its own kind, within its own circuit
);

COMMENT ON TABLE gold_qasmbench_register IS
    'One row represents one declared quantum or classical register of one circuit variant.';

CREATE INDEX IF NOT EXISTS idx_qasmbench_register_circuit_id ON gold_qasmbench_register(circuit_id);

CREATE TABLE IF NOT EXISTS gold_qasmbench_stabilizer_check ( -- one row per detected parity-check relationship
    source_record_id  TEXT PRIMARY KEY, -- traces which Silver row this Gold row came from
    circuit_id         TEXT NOT NULL REFERENCES gold_qasmbench_circuit(circuit_id) ON DELETE CASCADE, -- which circuit variant this parity check appears in
    check_id            TEXT NOT NULL, -- "<ancilla_qubit>#<occurrence>", unique within its circuit
    ancilla_qubit       TEXT NOT NULL, -- the qubit collecting the parity and being measured into syndrome_bit
    data_qubits         TEXT[] NOT NULL CHECK (cardinality(data_qubits) >= 1), -- the data qubits whose parity this check measures
    syndrome_bit         TEXT NOT NULL, -- the classical bit the ancilla is measured into

    UNIQUE (circuit_id, check_id) -- a check_id is only unique within its own circuit
);

COMMENT ON TABLE gold_qasmbench_stabilizer_check IS
    'One row represents one explicit parity-check relationship (ancilla, its data qubits, and the resulting syndrome bit) found in one circuit variant.';

CREATE INDEX IF NOT EXISTS idx_qasmbench_stabilizer_check_circuit_id ON gold_qasmbench_stabilizer_check(circuit_id);

CREATE TABLE IF NOT EXISTS gold_qasmbench_conditional_correction ( -- one row per syndrome-conditioned correction gate
    source_record_id    TEXT PRIMARY KEY, -- traces which Silver row this Gold row came from
    circuit_id            TEXT NOT NULL REFERENCES gold_qasmbench_circuit(circuit_id) ON DELETE CASCADE, -- which circuit variant this correction appears in
    condition_register    TEXT NOT NULL, -- the classical register the if(...) condition reads
    condition_value        BIGINT NOT NULL CHECK (condition_value >= 0), -- the syndrome value that triggers this correction
    gate                    TEXT NOT NULL, -- the single-qubit gate applied when the condition holds
    target_qubit             TEXT NOT NULL -- which qubit the gate is applied to
);

COMMENT ON TABLE gold_qasmbench_conditional_correction IS
    'One row represents one syndrome-conditioned correction gate applied to one target qubit in one circuit variant.';

CREATE INDEX IF NOT EXISTS idx_qasmbench_conditional_correction_circuit_id ON gold_qasmbench_conditional_correction(circuit_id);
"""


def _register_rows(circuit_rows: list[dict]) -> list[tuple[str, str, str, int]]:
    """Decompose every circuit row's register_declarations JSON text into one
    tuple per declared register - the concrete step behind the
    register-decomposition decision above."""
    rows: list[tuple[str, str, str, int]] = []
    for row in circuit_rows:
        declarations = json.loads(row["register_declarations"])
        for kind, key in (("qreg", "qregs"), ("creg", "cregs")):
            for register in declarations.get(key, []):
                rows.append((row["circuit_id"], kind, register["name"], register["size"]))
    return rows


def _check_repetition_code_labels(circuit_rows: list[dict], stabilizer_rows: list[dict]) -> None:
    """A distance-d repetition code has d data qubits checked by d-1
    neighbouring parity checks. Raise if a circuit labelled as one in
    QASMBENCH_CODE_LABELS does not have that structure in Silver, so a
    hand-written label can never silently disagree with the parsed data."""
    for row in circuit_rows:
        label = QASMBENCH_CODE_LABELS.get(row["benchmark_name"])
        if label is None or label[0] != "repetition_code":
            continue
        distance = label[1]
        checks = [c for c in stabilizer_rows if c["circuit_id"] == row["circuit_id"]]
        data_qubits = {q for c in checks for q in c["data_qubits"]}
        if len(checks) != distance - 1 or len(data_qubits) != distance:
            raise ValueError(
                f"{row['circuit_id']} is labelled repetition_code distance {distance}, but Silver has "
                f"{len(checks)} parity checks over {len(data_qubits)} data qubits "
                f"(expected {distance - 1} checks over {distance} data qubits)"
            )


def load_qasmbench_gold(conn, settings) -> tuple[int, int, int, int, str, str, str]:
    """Load Silver's qasmbench Parquet tables into Postgres, all-or-nothing.

    Returns (circuit count, register count, stabilizer-check count, correction count,
    circuit hash, stabilizer hash, correction hash).

    Truncates gold_qasmbench_circuit first (CASCADE also clears
    gold_qasmbench_register, gold_qasmbench_stabilizer_check, and
    gold_qasmbench_conditional_correction, since they all reference it), then
    inserts fresh - so a rerun never duplicates rows, and if anything fails
    partway, the transaction rolls back and the previous complete data stays
    in place. Mirrors load_google_gold's shape: reads its own Silver files
    (with hashes), truncates, inserts, and returns counts + hashes for the
    caller to reconcile and record.
    """
    circuit_table, circuit_hash = _read_silver_parquet_with_hash(settings, "silver/qasmbench/circuit.parquet")
    stabilizer_table, stabilizer_hash = _read_silver_parquet_with_hash(
        settings, "silver/qasmbench/stabilizer_check.parquet"
    )
    correction_table, correction_hash = _read_silver_parquet_with_hash(
        settings, "silver/qasmbench/conditional_correction.parquet"
    )

    circuit_rows = circuit_table.to_pylist()
    stabilizer_rows = stabilizer_table.to_pylist()
    correction_rows = correction_table.to_pylist()
    register_rows = _register_rows(circuit_rows)
    _check_repetition_code_labels(circuit_rows, stabilizer_rows)

    with conn.cursor() as cur:
        # Truncating the parent table cascades to the three dependent tables.
        cur.execute("TRUNCATE TABLE gold_qasmbench_circuit RESTART IDENTITY CASCADE;")

        cur.executemany(
            """
            INSERT INTO gold_qasmbench_circuit
                (circuit_id, source_record_id, benchmark_name, variant,
                 qubit_count, measurement_count, two_qubit_gate_count, code_family, distance)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (r["circuit_id"], r["source_record_id"], r["benchmark_name"], r["variant"],
                 r["qubit_count"], r["measurement_count"], r["two_qubit_gate_count"],
                 *QASMBENCH_CODE_LABELS.get(r["benchmark_name"], (None, None)))
                for r in circuit_rows
            ],
        )

        cur.executemany(
            """
            INSERT INTO gold_qasmbench_register
                (circuit_id, register_kind, register_name, register_size)
            VALUES (%s, %s, %s, %s)
            """,
            register_rows,
        )

        cur.executemany(
            """
            INSERT INTO gold_qasmbench_stabilizer_check
                (source_record_id, circuit_id, check_id, ancilla_qubit, data_qubits, syndrome_bit)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (r["source_record_id"], r["circuit_id"], r["check_id"], r["ancilla_qubit"],
                 list(r["data_qubits"]), r["syndrome_bit"])
                for r in stabilizer_rows
            ],
        )

        cur.executemany(
            """
            INSERT INTO gold_qasmbench_conditional_correction
                (source_record_id, circuit_id, condition_register, condition_value, gate, target_qubit)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (r["source_record_id"], r["circuit_id"], r["condition_register"],
                 r["condition_value"], r["gate"], r["target_qubit"])
                for r in correction_rows
            ],
        )

    return (
        len(circuit_rows), len(register_rows), len(stabilizer_rows), len(correction_rows),
        circuit_hash, stabilizer_hash, correction_hash,
    )


def _load_qasmbench(conn, settings) -> dict:
    """Load qasmbench into Gold and check that every Silver row arrived exactly once.

    Same shape as _load_google: run the schema DDL, load, then re-query the
    actual Gold tables and raise if they disagree with what the loader
    returned - so a silent drop rolls back the whole run() transaction
    instead of being recorded as a false "match".
    """
    with conn.cursor() as cur:
        cur.execute(QASMBENCH_GOLD_SCHEMA_SQL)

    (circuit_count, register_count, check_count, correction_count,
     circuit_hash, stabilizer_hash, correction_hash) = load_qasmbench_gold(conn, settings)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_circuit;")
        gold_circuit_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_register;")
        gold_register_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_stabilizer_check;")
        gold_check_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM gold_qasmbench_conditional_correction;")
        gold_correction_count = cur.fetchone()[0]
        if (gold_circuit_count, gold_register_count, gold_check_count, gold_correction_count) != (
            circuit_count, register_count, check_count, correction_count
        ):
            raise RuntimeError(
                f"Gold row counts do not match Silver: "
                f"circuit {gold_circuit_count} vs {circuit_count}, "
                f"register {gold_register_count} vs {register_count}, "
                f"stabilizer_check {gold_check_count} vs {check_count}, "
                f"conditional_correction {gold_correction_count} vs {correction_count}"
            )

    return {
        "circuit_count": circuit_count,
        "register_count": register_count,
        "check_count": check_count,
        "correction_count": correction_count,
        "circuit_hash": circuit_hash,
        "stabilizer_hash": stabilizer_hash,
        "correction_hash": correction_hash,
        "gold_circuit_count": gold_circuit_count,
        "gold_register_count": gold_register_count,
        "gold_check_count": gold_check_count,
        "gold_correction_count": gold_correction_count,
    }


# ---------------------------------------------------------------------------
# run(): loads every source into Gold as one all-or-nothing update
# ---------------------------------------------------------------------------

# Every source's Gold tables live in one dedicated schema (team agreement), not in
# the default "public" schema. Tests point this at a throwaway schema instead.

GOLD_SCHEMA = "gold"

# One row per experiment (google_qec, qec_syndromes) or circuit variant
# (qasmbench), in the shared vocabulary. UNION ALL, never JOIN: the sources
# sit side by side under the same column names without any row being matched
# to another source's row. It also makes the rejected relationship visible:
# qasmbench's distance-3 circuit is a repetition_code, Google's and the
# simulated distance-3 experiments are surface_code.
QEC_CATALOG_VIEW_SQL = """
CREATE OR REPLACE VIEW v_qec_experiment_catalog AS
SELECT 'google_qec'::TEXT AS source_name, 'experiment'::TEXT AS entity_kind, experiment_id AS entity_id,
       code_family, distance, rounds
FROM gold_google_experiment
UNION ALL
SELECT 'qec_syndromes', 'experiment', experiment_id,
       code_family, distance, round_count
FROM gold_syndrome_experiment
UNION ALL
SELECT 'qasmbench', 'circuit', circuit_id,
       code_family, distance, NULL::INTEGER -- a circuit definition has no executed QEC rounds
FROM gold_qasmbench_circuit;

COMMENT ON VIEW v_qec_experiment_catalog IS
    'One row represents one experiment (google_qec, qec_syndromes) or circuit variant (qasmbench), listed in the shared QEC vocabulary; sources are stacked, never joined.';
"""


ML_VIEWS_SQL = """
CREATE OR REPLACE VIEW v_ml_syndrome_decoder_example AS
SELECT
    o.observation_id                  AS example_id,
    o.experiment_id,
    e.physical_fault_rate::FLOAT8     AS physical_fault_rate,
    o.syndrome_bits,
    e.round_count,
    e.check_count,
    o.logical_error_label,
    o.quantity                        AS sample_weight
FROM gold_syndrome_observation o
JOIN gold_syndrome_experiment e ON e.experiment_id = o.experiment_id
ORDER BY o.experiment_id, o.observation_id;

COMMENT ON VIEW v_ml_syndrome_decoder_example IS
    'One row represents one distinct syndrome-and-label observation within one physical-fault-rate experiment (ml_syndrome_decoder_example, without data_split).';

CREATE OR REPLACE VIEW v_ml_google_decoder_example AS
SELECT
    LEFT(ENCODE(SHA256(CONVERT_TO('google_qec:' || s.experiment_id || ':' || s.shot_index, 'UTF8')), 'hex'), 32)
                                      AS example_id,
    s.experiment_id,
    s.shot_index::BIGINT              AS shot_index,
    e.distance,
    e.rounds,
    e.center_row,
    e.center_col,
    e.detector_count,
    s.detector_event_count,
    s.detector_bits,
    BOOL_OR(p.predicted_flip) FILTER (WHERE p.decoder_name = 'belief_matching')            AS belief_matching_prediction,
    BOOL_OR(p.predicted_flip) FILTER (WHERE p.decoder_name = 'correlated_matching')        AS correlated_matching_prediction,
    BOOL_OR(p.predicted_flip) FILTER (WHERE p.decoder_name = 'pymatching')                 AS pymatching_prediction,
    BOOL_OR(p.predicted_flip) FILTER (WHERE p.decoder_name = 'tensor_network_contraction') AS tensor_network_contraction_prediction,
    s.actual_observable_flip
FROM gold_google_shot s
JOIN gold_google_experiment e ON e.experiment_id = s.experiment_id
LEFT JOIN gold_google_decoder_prediction p
       ON p.experiment_id = s.experiment_id AND p.shot_index = s.shot_index
GROUP BY s.experiment_id, s.shot_index, e.experiment_id
ORDER BY s.experiment_id, s.shot_index;

COMMENT ON VIEW v_ml_google_decoder_example IS
    'One row represents one hardware shot in one Google surface-code experiment (ml_google_decoder_example, without data_split).';
"""


def _load_google(conn, settings) -> dict:
    """Load google_qec into Gold and check that every Silver row arrived exactly once.

    Returns the counts and hashes that run() writes to the results files.
    """
    with conn.cursor() as cur:
        cur.execute(GOOGLE_GOLD_SCHEMA_SQL)

    exp_count, shot_count, decoder_prediction_count, experiment_hash, shot_hash = load_google_gold(conn, settings)

    #checking the total counts -> if there is an error -> will trigger mismatch rollback
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT (*) FROM gold_google_experiment;")
        gold_exp_count = cur.fetchone()[0] # cur.fetchone()[0] retrieves the first result row as a tuple (e.g., (150,)) and extracts the single value inside it (150).
        cur.execute ("SELECT COUNT (*) FROM gold_google_shot;") #for shot count 
        gold_shot_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM gold_google_decoder_prediction;") #for decoder count 
        gold_decoder_prediction_count = cur.fetchone()[0]
        #checking and comparing if there is any mismatch in the total count 
        if (gold_exp_count, gold_shot_count, gold_decoder_prediction_count) != (exp_count, shot_count, decoder_prediction_count):
            raise RuntimeError(
                f"Gold row counts do not match Silver: "
                f"experiment {gold_exp_count} vs {exp_count}, "
                f"shot {gold_shot_count} vs {shot_count}, "
                f"decoder_prediction {gold_decoder_prediction_count} vs {decoder_prediction_count}"
            )

    return {
        "exp_count": exp_count,
        "shot_count": shot_count,
        "decoder_prediction_count": decoder_prediction_count,
        "experiment_hash": experiment_hash,
        "shot_hash": shot_hash,
        "gold_exp_count": gold_exp_count,
        "gold_shot_count": gold_shot_count,
        "gold_decoder_prediction_count": gold_decoder_prediction_count,
    }


def _load_syndromes(conn, settings) -> dict:
    """Load qec_syndromes into Gold and check that every Silver row and its
    quantity weight arrived exactly once.

    Returns the counts and hash that run() writes to the results files.
    """
    syndromes_table, input_hash = _read_silver_parquet_with_hash(settings, SILVER_SYNDROMES_PATH)

    silver_rows = syndromes_table.to_pylist()
    silver_quantity_by_experiment: dict[str, int] = {}
    for r in silver_rows:
        silver_quantity_by_experiment[r["experiment_id"]] = (
            silver_quantity_by_experiment.get(r["experiment_id"], 0) + r["quantity"]
        )

    with conn.cursor() as cursor:
        create_syndrome_tables(cursor)
        load_syndrome_tables(cursor, syndromes_table)

        cursor.execute("SELECT COUNT(*) FROM gold_syndrome_experiment")
        experiment_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM gold_syndrome_pattern")
        pattern_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM gold_syndrome_observation")
        observation_count = cursor.fetchone()[0]
        cursor.execute("SELECT COALESCE(SUM(quantity), 0) FROM gold_syndrome_observation")
        # SUM(quantity) over a BIGINT column comes back as NUMERIC (Decimal in Python), not int -- json.dumps can't serialize Decimal.
        gold_quantity_total = int(cursor.fetchone()[0])
        cursor.execute(
            "SELECT experiment_id, COALESCE(SUM(quantity), 0) FROM gold_syndrome_observation GROUP BY experiment_id"
        )
        gold_quantity_by_experiment = {
            experiment_id: int(total) for experiment_id, total in cursor.fetchall()
        }

        # every Silver row (and its quantity weight) must arrive in Gold exactly once. Raising here, before the commit, rolls the whole load back, so
        # a load that lost or duplicated rows never replaces the previous Gold data
        rows_match = observation_count == syndromes_table.num_rows
        quantity_match = gold_quantity_by_experiment == silver_quantity_by_experiment
        if not (rows_match and quantity_match):
            raise ValueError(
                f"Silver-to-Gold mismatch: {syndromes_table.num_rows} Silver rows vs "
                f"{observation_count} Gold rows; quantity per experiment "
                f"Silver={silver_quantity_by_experiment} Gold={gold_quantity_by_experiment}"
            )

    return {
        "input_hash": input_hash,
        "silver_rows": syndromes_table.num_rows,
        "experiment_count": experiment_count,
        "pattern_count": pattern_count,
        "observation_count": observation_count,
        "rows_match": rows_match,
        "silver_quantity_by_experiment": silver_quantity_by_experiment,
        "gold_quantity_total": gold_quantity_total,
        "gold_quantity_by_experiment": gold_quantity_by_experiment,
        "quantity_match": quantity_match,
    }


def run(run_id: str) -> StageResult:
    """Load every source's Silver tables into Gold PostgreSQL as one all-or-nothing update.

    google_qec, qec_syndromes, and qasmbench are all loaded in the same
    transaction: if any source fails (a constraint, or a Silver/Gold count
    mismatch), nothing is committed and the previous complete Gold data
    stays in place.
    """
    settings = Settings.from_environment()
    result = StageResult(stage="load_postgres", run_id=run_id)

    conn = connections.postgres_connection(settings)
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}")
            cur.execute(f"SET search_path TO {GOLD_SCHEMA}")

        google = _load_google(conn, settings)
        syndromes = _load_syndromes(conn, settings)
        qasmbench = _load_qasmbench(conn, settings)

        with conn.cursor() as cur:
            cur.execute(QEC_CATALOG_VIEW_SQL)
            cur.execute(ML_VIEWS_SQL)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    #input from silver stage (google experiment + shot, syndrome observations, qasmbench circuits/checks/corrections)
    result.input_count = (
        google["exp_count"] + google["shot_count"] + syndromes["silver_rows"]
        + qasmbench["circuit_count"] + qasmbench["check_count"] + qasmbench["correction_count"]
    )
    #output from gold stage (google experiment + shot, syndrome observations, qasmbench circuits/registers/checks/corrections)
    result.output_count = (
        google["gold_exp_count"] + google["gold_shot_count"] + syndromes["observation_count"]
        + qasmbench["gold_circuit_count"] + qasmbench["gold_register_count"]
        + qasmbench["gold_check_count"] + qasmbench["gold_correction_count"]
    )
    result.issue_count = 0
    result.finish()

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)

    # --- google_qec evidence ---
    expected_decoder_prediction_count = google["shot_count"] * len(DECODER_NAMES)

    merge_json_file(
        RESULTS_PART1 / "run.json",
        "load_postgres", "google_qec",
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hashes": {
                "silver/google_qec/experiment.parquet": google["experiment_hash"],
                "silver/google_qec/shot.parquet": google["shot_hash"],
            },
            "output_row_counts": {
                "gold_google_experiment": google["exp_count"],
                "gold_google_shot": google["shot_count"],
                "gold_google_decoder_prediction": google["decoder_prediction_count"],
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "load_postgres", "google_qec",
        payload={
            "experiment": {"silver_rows": google["exp_count"], "gold_loaded": google["gold_exp_count"], "match": google["gold_exp_count"] == google["exp_count"]},
            "shot": {"silver_rows": google["shot_count"], "gold_loaded": google["gold_shot_count"], "match": google["gold_shot_count"] == google["shot_count"]},
            "decoder_prediction": {"expected": expected_decoder_prediction_count, "gold_loaded": google["gold_decoder_prediction_count"], "match": google["gold_decoder_prediction_count"] == expected_decoder_prediction_count},
        },
    )

    # --- qec_syndromes evidence ---
    silver_quantity_by_experiment = syndromes["silver_quantity_by_experiment"]
    gold_quantity_by_experiment = syndromes["gold_quantity_by_experiment"]
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "load_postgres",
        SYNDROMES_SOURCE_NAME,
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hash": syndromes["input_hash"],
            "output_row_counts": {
                "gold_syndrome_experiment": syndromes["experiment_count"],
                "gold_syndrome_pattern": syndromes["pattern_count"],
                "gold_syndrome_observation": syndromes["observation_count"],
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "load_postgres",
        SYNDROMES_SOURCE_NAME,
        payload={
            "silver_rows": syndromes["silver_rows"],
            "gold_observation_rows": syndromes["observation_count"],
            "rows_match": syndromes["rows_match"],
            "silver_quantity_total": sum(silver_quantity_by_experiment.values()),
            "gold_quantity_total": syndromes["gold_quantity_total"],
            "quantity_match": syndromes["quantity_match"],
            "quantity_by_experiment": {
                experiment_id: {
                    "silver": silver_quantity_by_experiment.get(experiment_id, 0),
                    "gold": gold_quantity_by_experiment.get(experiment_id, 0),
                    "match": silver_quantity_by_experiment.get(experiment_id, 0)
                    == gold_quantity_by_experiment.get(experiment_id, 0),
                }
                for experiment_id in silver_quantity_by_experiment
            },
        },
    )

    # --- qasmbench evidence ---
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "load_postgres", QASMBENCH_SOURCE_NAME,
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hashes": {
                "silver/qasmbench/circuit.parquet": qasmbench["circuit_hash"],
                "silver/qasmbench/stabilizer_check.parquet": qasmbench["stabilizer_hash"],
                "silver/qasmbench/conditional_correction.parquet": qasmbench["correction_hash"],
            },
            "output_row_counts": {
                "gold_qasmbench_circuit": qasmbench["circuit_count"],
                "gold_qasmbench_register": qasmbench["register_count"],
                "gold_qasmbench_stabilizer_check": qasmbench["check_count"],
                "gold_qasmbench_conditional_correction": qasmbench["correction_count"],
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "load_postgres", QASMBENCH_SOURCE_NAME,
        payload={
            "circuit": {"silver_rows": qasmbench["circuit_count"], "gold_loaded": qasmbench["gold_circuit_count"], "match": qasmbench["gold_circuit_count"] == qasmbench["circuit_count"]},
            "register": {"silver_rows": qasmbench["register_count"], "gold_loaded": qasmbench["gold_register_count"], "match": qasmbench["gold_register_count"] == qasmbench["register_count"]},
            "stabilizer_check": {"silver_rows": qasmbench["check_count"], "gold_loaded": qasmbench["gold_check_count"], "match": qasmbench["gold_check_count"] == qasmbench["check_count"]},
            "conditional_correction": {"silver_rows": qasmbench["correction_count"], "gold_loaded": qasmbench["gold_correction_count"], "match": qasmbench["gold_correction_count"] == qasmbench["correction_count"]},
        },
    )

    return result
