'''
This code file is for the integration of relational database 

According to the rubrics, it includes:
1) creation of the user defined tables
2) primary and foreign keys
3) includes value checks and indexes

Note: Repeated loads must not create duplicate records
'''
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

# The DDL that creates both Gold tables: gold_google_experiment and gold_google_shot
# Using IF NOT EXISTS to carry out no operations if the table already exists -> Safer choice 
#gold_google_experiment has one row for one experiment, whereas gold_google_shot has many rows for a single experiment
GOOGLE_GOLD_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gold_google_experiment ( --for each experiment directory
    experiment_id       TEXT PRIMARY KEY, -- one row per experiment
    basis                TEXT NOT NULL, --measurement basis
    distance             INTEGER NOT NULL CHECK (distance > 0), --the distance must be positive 
    rounds                INTEGER NOT NULL CHECK (rounds > 0), -- QEC rounds must be positive
    shots                 INTEGER NOT NULL CHECK (shots > 0), --shot count must be positive
    center_row            INTEGER NOT NULL, -- processor location, row coordinate
    center_col            INTEGER NOT NULL, --processor location, column coordinate
    measurement_count     INTEGER NOT NULL CHECK (measurement_count > 0), --bits per shot in measurements.b8
    detector_count        INTEGER NOT NULL CHECK (detector_count > 0) --bits per shot in detection_events.b8 
);
COMMENT ON TABLE gold_google_experiment IS
    'One row represents one Google hardware experiment configuration (a distance/rounds/processor-location combination).';

CREATE TABLE IF NOT EXISTS gold_google_shot ( -- for the single shots
    experiment_id                          TEXT NOT NULL REFERENCES gold_google_experiment(experiment_id), 
    shot_index                             INTEGER NOT NULL CHECK (shot_index >= 0), -- can be zero based and non negative
    measurement_bits                       BYTEA NOT NULL, --raw packed bits for this shot
    sweep_bits                             BYTEA NOT NULL, -- configuration bits for this shot
    detector_bits                          BYTEA NOT NULL, -- derived detector events for this shot
    detector_event_count                   INTEGER NOT NULL CHECK (detector_event_count >= 0), --counts the set of detector bits
    actual_observable_flip                 BOOLEAN NOT NULL, --ground truth outcome 
    belief_matching_prediction             BOOLEAN NOT NULL, -- decoder 1's prediction
    correlated_matching_prediction         BOOLEAN NOT NULL, --decoder 2's prediction
    pymatching_prediction                  BOOLEAN NOT NULL, --decoder 3's prediction
    tensor_network_contraction_prediction  BOOLEAN NOT NULL, --decoder 4's prediction
    PRIMARY KEY (experiment_id, shot_index)  -- composite: a shot is only unique WITHIN its experiment
);
COMMENT ON TABLE gold_google_shot IS
    'One row represents one hardware shot (one circuit execution) within one Google experiment, with its raw bits, actual outcome, and all four decoder predictions.';

CREATE INDEX IF NOT EXISTS idx_google_shot_experiment_id ON gold_google_shot(experiment_id);
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

    
def load_google_gold(conn, settings) -> tuple[int, int]: #this function loads the silver data to gold
    # Args:
    # conn: takes the PostgreSQL connection, settings = env config
     # returns a tuple of 2 integers (experiment count, shot count).
    
    """Load Silver's google_qec Parquet tables into Postgres, all-or-nothing.

    Truncates both tables first, then inserts fresh - so a rerun never
    duplicates rows, and if anything fails partway, the transaction rolls
    back and the previous complete data stays in place.
    """
    experiment_table = _read_silver_parquet(settings, "silver/google_qec/experiment.parquet") #calling the parquet tables for experiments
    shot_table = _read_silver_parquet(settings, "silver/google_qec/shot.parquet") #calling the parquet tables for the shots

    with conn.cursor() as cur: # Open a database cursor to run SQL commands.
        cur.execute("TRUNCATE TABLE gold_google_shot, gold_google_experiment RESTART IDENTITY CASCADE;") #using TRUNCATE as its faster than DELETE

        cur.executemany( ## executemany runs the same SQL statement many times with different values
            
            """
            INSERT INTO gold_google_experiment
                (experiment_id, basis, distance, rounds, shots, center_row, center_col, measurement_count, detector_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            # contains the template for insertion of values 
            [
                (r["experiment_id"], r["basis"], r["distance"], r["rounds"], r["shots"],
                 r["center_row"], r["center_col"], r["measurement_count"], r["detector_count"])
                for r in experiment_table.to_pylist()
            ], # Build a list of tuples, one per experiment row.  Each tuple's values match the %s placeholders in order.

             # experiment_table.to_pylist() converts the pyarrow table into a list of Python dicts (one dict per row) that we can pull values from by column name.
        )

        cur.executemany(
            """
            INSERT INTO gold_google_shot
                (experiment_id, shot_index, measurement_bits, sweep_bits, detector_bits,
                 detector_event_count, actual_observable_flip,
                 belief_matching_prediction, correlated_matching_prediction,
                 pymatching_prediction, tensor_network_contraction_prediction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            #11 placehodlers for the shot record 
            [
                (r["experiment_id"], r["shot_index"], r["measurement_bits"], r["sweep_bits"], r["detector_bits"],
                 r["detector_event_count"], r["actual_observable_flip"],
                 r["belief_matching_prediction"], r["correlated_matching_prediction"],
                 r["pymatching_prediction"], r["tensor_network_contraction_prediction"])
                for r in shot_table.to_pylist()
            ],
        )

    return experiment_table.num_rows, shot_table.num_rows

from quantum_lake_student.models import RESULTS_PART1, StageResult, merge_json_file, code_revision


def run(run_id: str) -> StageResult:
    """Load Silver's google_qec tables into Gold PostgreSQL, all-or-nothing."""
    settings = Settings.from_environment()
    result = StageResult(stage="load_postgres", run_id=run_id)

    conn = connections.postgres_connection(settings)
    try:
        with conn.cursor() as cur:
            cur.execute(GOOGLE_GOLD_SCHEMA_SQL)

        exp_count, shot_count = load_google_gold(conn, settings)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    result.input_count = exp_count + shot_count
    result.output_count = exp_count + shot_count
    result.issue_count = 0
    result.finish()

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)
    merge_json_file(
        RESULTS_PART1 / "run.json",
        "load_postgres", "google_qec",
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "output_row_counts": {
                "gold_google_experiment": exp_count,
                "gold_google_shot": shot_count,
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "load_postgres", "google_qec",
        payload={
            "experiment": {"loaded": exp_count},
            "shot": {"loaded": shot_count},
        },
    )

    return result