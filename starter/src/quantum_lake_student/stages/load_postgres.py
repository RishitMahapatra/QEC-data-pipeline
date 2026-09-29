'''
This code file is for the integration of relational database 

According to the rubrics, it includes:
1) creation of the user defined tables
2) primary and foreign keys
3) includes value checks and indexes

Note: Repeated loads must not create duplicate records
'''

# Detector storage decision: detector_bits is stored as one packed BYTEA value per shot row,
# not one row per individual detector (200 detectors = 25 bytes for distance 3, 600 = 75 bytes
# for distance 5). Full reasoning and measured sizes: reports/gold_stage/gold_stage_google_qec.md

import io #used to read the raw MinIO bytes so pyarrow can read them as a file

import pyarrow as pa  # in-memory tabular data type used by Silver
import pyarrow.parquet as pq  # reads Parquet files into pyarrow tables

from quantum_lake_student import connections # for postgres_connection() helper
from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client #for creation of minio client
from quantum_lake_student.models import RESULTS_PART1, StageResult, merge_json_file, code_revision

'''
For the google_qec gold stage -> takes the two validated silver parquet tables. Since Parquet tables are for indexing and storage, running queries is not handled. For that reason, we implement PostgreSQL at the gold stage. Takes the two validated Silver Parquet tables
(experiment.parquet, shot.parquet) and loads them into real PostgreSQL tables that enforce keys, foreign keys, and CHECK constraints.

Load is all-or-nothing (rollback on any failure), and idempotent (rerun replaces the rows, doesn't duplicate).
'''

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
    experiment_id                          TEXT NOT NULL REFERENCES gold_google_experiment(experiment_id), 
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
    decoder_name TEXT PRIMARY KEY
);

COMMENT ON TABLE gold_google_decoder IS
    'One row per decoder algorithm available for google_qec (belief_matching, correlated_matching, pymatching, tensor_network_contraction).';

-- One row per shot per decoder, separating "what it guessed" from "whether it was wrong"

CREATE TABLE IF NOT EXISTS gold_google_decoder_prediction (
    experiment_id     TEXT NOT NULL,
    shot_index        INTEGER NOT NULL,
    decoder_name      TEXT NOT NULL REFERENCES gold_google_decoder(decoder_name),
    predicted_flip    BOOLEAN NOT NULL,
    was_wrong         BOOLEAN NOT NULL,
    PRIMARY KEY (experiment_id, shot_index, decoder_name),
    FOREIGN KEY (experiment_id, shot_index) REFERENCES gold_google_shot(experiment_id, shot_index)
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
    import hashlib #this is the hash library 
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
                (experiment_id, source_record_id, basis, distance, rounds, shots, center_row, center_col, measurement_count, detector_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            # contains the template for insertion of values 
            [
                (r["experiment_id"], r["source_record_id"], r["basis"], r["distance"], r["rounds"], r["shots"],
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

def run(run_id: str) -> StageResult:
    """Load Silver's google_qec tables into Gold PostgreSQL, all-or-nothing."""
    settings = Settings.from_environment()
    result = StageResult(stage="load_postgres", run_id=run_id)

    conn = connections.postgres_connection(settings)
    try:
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
                
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    #input from silver stage (experiment + shot)
    result.input_count = exp_count + shot_count
    #output from gold stage (experiment + shot)
    result.output_count = gold_exp_count + gold_shot_count
    result.issue_count = 0
    result.finish()

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)

    expected_decoder_prediction_count = shot_count * len(DECODER_NAMES)

    merge_json_file(
        RESULTS_PART1 / "run.json",
        "load_postgres", "google_qec",
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_hashes": {
                "silver/google_qec/experiment.parquet": experiment_hash,
                "silver/google_qec/shot.parquet": shot_hash,
            },
            "output_row_counts": {
                "gold_google_experiment": exp_count,
                "gold_google_shot": shot_count,
                "gold_google_decoder_prediction": decoder_prediction_count,
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "load_postgres", "google_qec",
        payload={
            "experiment": {"silver_rows": exp_count, "gold_loaded": gold_exp_count, "match": gold_exp_count == exp_count},
            "shot": {"silver_rows": shot_count, "gold_loaded": gold_shot_count, "match": gold_shot_count == shot_count},
            "decoder_prediction": {"expected": expected_decoder_prediction_count, "gold_loaded": gold_decoder_prediction_count, "match": gold_decoder_prediction_count == expected_decoder_prediction_count},
        },
    )

    return result