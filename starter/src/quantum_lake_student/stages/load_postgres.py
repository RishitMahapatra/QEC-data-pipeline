'''
This code file is for the integration of relational database

According to the rubrics, it includes:
1) creation of the user defined tables
2) primary and foreign keys
3) includes value checks and indexes

Note: Repeated loads must not create duplicate records
'''

# Detector storage decision (google_qec): detector_bits is stored as one packed BYTEA value per shot row,
# not one row per individual detector. Distance-3 has 200 detectors/shot (25 bytes packed),
# distance-5 has 600 (75 bytes). Across all 250,000 shots that is
# (200,000 x 25) + (50,000 x 75) = 8,750,000 bytes. One row per detector would instead need
# (200 x 200,000) + (600 x 50,000) = 70,000,000 rows (~280x more), for data that no required
# query or ML table needs at single-detector granularity.

# qasmbench does not have this problem: the whole archive is 3 benchmarks x 2 variants = 6
# circuits, with at most a handful of parity checks/corrections per circuit. Nothing here is
# large enough to need a packing decision; the design questions for qasmbench are instead about
# *shape* (see the two comment blocks right above QASMBENCH_GOLD_SCHEMA_SQL below).

import io  # used to read the raw MinIO bytes so pyarrow can read them as a file
import json  # qasmbench's register_declarations Silver column is JSON text, decoded here

import pyarrow as pa  # in-memory tabular data type used by Silver
import pyarrow.parquet as pq  # reads Parquet files into pyarrow tables

from quantum_lake_student import connections  # for connections.postgres_connection() / minio_client()
from quantum_lake_student.config import Settings
from quantum_lake_student.models import RESULTS_PART1, StageResult, merge_json_file, code_revision

'''
For the google_qec gold stage -> takes the two validated silver parquet tables. Since Parquet tables are for indexing and storage, running queries is not handled. For that reason, we implement PostgreSQL at the gold stage. Takes the two validated Silver Parquet tables
(experiment.parquet, shot.parquet) and loads them into real PostgreSQL tables that enforce keys, foreign keys, and CHECK constraints.

Load is all-or-nothing (rollback on any failure), and idempotent (rerun replaces the rows, doesn't duplicate).

For the qasmbench gold stage -> takes the three validated silver parquet tables
(circuit.parquet, stabilizer_check.parquet, conditional_correction.parquet) and loads them the
same way: real PostgreSQL tables with keys/foreign keys/CHECK constraints, an all-or-nothing
load, and an idempotent rerun. See the comment blocks above QASMBENCH_GOLD_SCHEMA_SQL for what
is different about qasmbench's shape compared to google_qec's, and why.

Each source is its own all-or-nothing unit (own schema DDL, own truncate-and-reload
transaction, own row_counts.json/run.json entry) - a failure loading one source does not touch
the other source's already-committed Gold tables. run() at the bottom just calls both and adds
up the totals, the same way prepare_data.py's Silver run() calls each source's own builder and
sums their counts.
'''

# ---------------------------------------------------------------------------
# Shared lake-read helper - both sources' Silver output is read the same way,
# so there is exactly one seam (`_read_silver_parquet`) that both load
# functions call and that tests monkeypatch to inject canned Silver data.
# ---------------------------------------------------------------------------


def _read_lake_object(settings: Settings, object_name: str) -> bytes:
    """Read one arbitrary object from the lake (MinIO bucket, or the local lake
    folder). Gold reads objects at the exact path Silver wrote them to, so unlike
    Bronze reads there is no bronze/raw prefix swap to do here."""
    if settings.lake_backend == "local":
        path = settings.local_lake_root / object_name
        if not path.exists():
            raise FileNotFoundError(f"Lake object not found: {path}")
        return path.read_bytes()
    client = connections.minio_client(settings)
    response = client.get_object(settings.s3_bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _read_silver_parquet(settings: Settings, object_name: str) -> pa.Table:
    """Read one Silver Parquet table from the lake as a pyarrow Table.

    Both google_qec's and qasmbench's load functions call this with their own
    object paths; it is the one function tests monkeypatch to hand a load
    function canned in-memory Silver rows instead of touching MinIO or a real
    local lake folder.
    """
    return pq.read_table(io.BytesIO(_read_lake_object(settings, object_name)))


# ---------------------------------------------------------------------------
# google_qec Gold: gold_google_experiment, gold_google_shot, gold_google_decoder,
# gold_google_decoder_prediction
# ---------------------------------------------------------------------------

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
    source_record_id TEXT NOT NULL, -- traces which Silver row this Gold row came from
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

    UNIQUE(distance, rounds, center_row, center_col) -- this ensures that no two experiments can share the same distance/rounds/location
);

COMMENT ON TABLE gold_google_experiment IS
    'One row represents one Google hardware experiment configuration (a distance/rounds/processor-location combination).';

CREATE TABLE IF NOT EXISTS gold_google_shot ( -- for the single shots
    experiment_id                          TEXT NOT NULL REFERENCES gold_google_experiment(experiment_id),
    source_record_id TEXT NOT NULL, -- traces which Silver row this Gold row came from
    shot_index                             INTEGER NOT NULL CHECK (shot_index >= 0), -- can be zero based and non negative
    measurement_bits                       BYTEA NOT NULL, --raw packed bits for this shot
    sweep_bits                             BYTEA NOT NULL, -- configuration bits for this shot
    detector_bits                          BYTEA NOT NULL CHECK (octet_length(detector_bits) IN (25, 75)), -- derived detector events; 25 bytes for d3, 75 bytes for d5

    detector_event_count                   INTEGER NOT NULL CHECK (detector_event_count >= 0), --counts the set of detector bits
    actual_observable_flip                 BOOLEAN NOT NULL, --ground truth outcome
    PRIMARY KEY (experiment_id, shot_index)  -- composite: a shot is only unique WITHIN its experiment
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


def load_google_gold(conn, settings) -> tuple[int, int, int]:  # this function loads the silver data to gold
    # Args:
    # conn: takes the PostgreSQL connection, settings = env config  returns a tuple of 3 integers (experiment count, shot count, decoder prediction count)

    """Load Silver's google_qec Parquet tables into Postgres, all-or-nothing.

    Truncates both tables first, then inserts fresh - so a rerun never
    duplicates rows, and if anything fails partway, the transaction rolls
    back and the previous complete data stays in place.
    """
    experiment_table = _read_silver_parquet(settings, "silver/google_qec/experiment.parquet")  # calling the parquet tables for experiments
    shot_table = _read_silver_parquet(settings, "silver/google_qec/shot.parquet")  # calling the parquet tables for the shots

    with conn.cursor() as cur:  # Open a database cursor to run SQL commands. TRUNCATE wipes all four Gold tables - CASCADE handles the dependency order automatically
        cur.execute(
            "TRUNCATE TABLE gold_google_decoder_prediction, gold_google_shot, "
            "gold_google_experiment, gold_google_decoder RESTART IDENTITY CASCADE;"
        )  # using TRUNCATE as its faster than DELETE

        cur.executemany(  # executemany runs the same SQL statement many times with different values
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
            ],  # Build a list of tuples, one per experiment row.  Each tuple's values match the %s placeholders in order.

            # experiment_table.to_pylist() converts the pyarrow table into a list of Python dicts (one dict per row) that we can pull values from by column name.
        )

        cur.executemany(
            """
            INSERT INTO gold_google_shot
                (experiment_id, source_record_id, shot_index, measurement_bits, sweep_bits, detector_bits,
                 detector_event_count, actual_observable_flip)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (r["experiment_id"], r["source_record_id"], r["shot_index"], r["measurement_bits"], r["sweep_bits"], r["detector_bits"],
                 r["detector_event_count"], r["actual_observable_flip"])
                for r in shot_table.to_pylist()
            ],
        )

        # Load the decoder table and the decoder_prediction table

        # 4 rows, one per decoder name
        cur.executemany(
            "INSERT INTO gold_google_decoder (decoder_name) VALUES (%s)",
            [(name,) for name in DECODER_NAMES],
        )

        # one row per shot per decoder: what it predicted, and whether that disagreed with the actual outcome - kept as two separate facts instead of mixing "guess" and "correctness" into one column
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

    return experiment_table.num_rows, shot_table.num_rows, len(decoder_prediction_rows)


def run_google_gold(settings: Settings, run_id: str) -> StageResult:
    """Load Silver's google_qec tables into Gold PostgreSQL, all-or-nothing.

    Opens one connection, runs the schema DDL, loads both tables, and lets a
    clean exit from the `with` block commit everything or an exception roll
    it all back (this is the same connection-as-context-manager behavior
    `run_qasmbench_gold` below relies on - see its comment for the exact
    commit/rollback semantics). Writes this source's row_counts.json/run.json
    entry, the same evidence pattern every Silver builder already uses.
    """
    result = StageResult(stage="load_postgres:google_qec", run_id=run_id)

    with connections.postgres_connection(settings) as connection:
        connection.execute(GOOGLE_GOLD_SCHEMA_SQL)
        exp_count, shot_count, decoder_prediction_count = load_google_gold(connection, settings)

    result.input_count = exp_count + shot_count
    result.output_count = exp_count + shot_count
    result.finish()

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)

    # expected decoder_prediction count is shot_count x number of decoders - used below to confirm nothing silently dropped
    expected_decoder_prediction_count = shot_count * len(DECODER_NAMES)

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
                "gold_google_decoder_prediction": decoder_prediction_count,
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "load_postgres", "google_qec",
        payload={
            # silver_rows and gold_loaded are the same read here since Gold never filters/rejects rows -
            # both are reported explicitly (rather than one number) so a future mismatch would be visible, not silent
            "experiment": {"silver_rows": exp_count, "gold_loaded": exp_count, "match": exp_count == exp_count},
            "shot": {"silver_rows": shot_count, "gold_loaded": shot_count, "match": shot_count == shot_count},
            "decoder_prediction": {"expected": expected_decoder_prediction_count, "gold_loaded": decoder_prediction_count, "match": decoder_prediction_count == expected_decoder_prediction_count},
        },
    )
    return result


# ---------------------------------------------------------------------------
# qasmbench Gold: gold_qasmbench_circuit, gold_qasmbench_register,
# gold_qasmbench_stabilizer_check, gold_qasmbench_conditional_correction
#
# Requirements this section is built against (assignment/data-sources.md's
# qasmbench section and assignment/brief.md's Gold section, not google_qec's
# shape): "at minimum, extract... parity-check ancilla, data participants,
# and syndrome bit where explicit; syndrome-controlled correction value,
# gate, and target" and "Keep QASMBench circuits as their own entities
# because no row-level match to the experiments is supplied." There is no
# google_qec-style "distance <-> detector_count" data-driven check here,
# because qasmbench has no equivalent cross-column fact to enforce - its
# data-driven rules are the two decisions below instead.
# ---------------------------------------------------------------------------

# Register-decomposition decision: Silver's circuit.parquet keeps
# register_declarations as one JSON-text column (a nested list of qreg/creg
# name+size pairs). Gold decomposes it into its own table,
# gold_qasmbench_register - one row per declared register - instead of also
# carrying a JSONB copy of the same column. Two representations of the same
# fact with nothing keeping them in sync is worse than one, and the brief
# says Gold must not "merely copy each Silver table into PostgreSQL": turning
# an opaque JSON blob into typed, constrained (register_size > 0,
# register_kind IN ('qreg','creg')) rows is exactly that kind of value-add.
# Only 4 registers per circuit variant (q, a, c, syn for the parity-check
# benchmark) - decomposing costs one small table, not real complexity.

# data_qubits-as-array decision: stabilizer_check.parquet's data_qubits is a
# small (2 elements), fixed-shape, ordered attribute of one check row, not an
# independent many-to-many relationship worth its own junction table - across
# the whole qasmbench archive it is ~8 total (qubit, check) pairs.
# gold_qasmbench_stabilizer_check.data_qubits stays a native TEXT[] column
# with a CHECK that it is never empty. Postgres arrays are queryable
# (= ANY(data_qubits), &&) without a fifth table for data this small.

QASMBENCH_GOLD_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gold_qasmbench_circuit ( -- one row per parsed QASMBench circuit variant
    circuit_id            TEXT PRIMARY KEY, -- "<benchmark>:<variant>", already unique and stable
    source_record_id      TEXT NOT NULL UNIQUE, -- traces which Silver row this Gold row came from
    benchmark_name         TEXT NOT NULL, -- which of the 3 QASMBench circuits this is
    variant                TEXT NOT NULL CHECK (variant IN ('source', 'transpiled')), -- as published, or after transpilation
    qubit_count            INTEGER NOT NULL CHECK (qubit_count > 0), -- declared qubits must be positive
    measurement_count      INTEGER NOT NULL CHECK (measurement_count >= 0), -- executed measure statements, expanded
    two_qubit_gate_count   INTEGER NOT NULL CHECK (two_qubit_gate_count >= 0), -- executed 2-qubit gates, expanded

    UNIQUE (benchmark_name, variant) -- a second uniqueness rule on the same fact as circuit_id, enforced independently
);

COMMENT ON TABLE gold_qasmbench_circuit IS
    'One row represents one parsed QASMBench circuit variant (a benchmark, as published or after transpilation).';

CREATE TABLE IF NOT EXISTS gold_qasmbench_register ( -- one row per declared qreg/creg of one circuit variant
    circuit_id      TEXT NOT NULL REFERENCES gold_qasmbench_circuit(circuit_id) ON DELETE CASCADE,
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
    circuit_id         TEXT NOT NULL REFERENCES gold_qasmbench_circuit(circuit_id) ON DELETE CASCADE,
    check_id            TEXT NOT NULL, -- "<ancilla_qubit>#<occurrence>", unique within its circuit
    ancilla_qubit       TEXT NOT NULL, -- the qubit collecting the parity and being measured into syndrome_bit
    data_qubits         TEXT[] NOT NULL CHECK (cardinality(data_qubits) >= 1), -- the data qubits whose parity this check measures; see the array-vs-junction-table decision above (cardinality(), not array_length(): array_length() of an empty array is NULL, not 0, and a NULL CHECK expression passes - cardinality() correctly returns 0)
    syndrome_bit         TEXT NOT NULL, -- the classical bit the ancilla is measured into

    UNIQUE (circuit_id, check_id)
);

COMMENT ON TABLE gold_qasmbench_stabilizer_check IS
    'One row represents one explicit parity-check relationship (ancilla, its data qubits, and the resulting syndrome bit) found in one circuit variant.';

CREATE INDEX IF NOT EXISTS idx_qasmbench_stabilizer_check_circuit_id ON gold_qasmbench_stabilizer_check(circuit_id);

CREATE TABLE IF NOT EXISTS gold_qasmbench_conditional_correction ( -- one row per syndrome-conditioned correction gate
    source_record_id    TEXT PRIMARY KEY, -- traces which Silver row this Gold row came from
    circuit_id            TEXT NOT NULL REFERENCES gold_qasmbench_circuit(circuit_id) ON DELETE CASCADE,
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


def load_qasmbench_gold(conn, settings) -> tuple[int, int, int, int]:
    # Args:
    # conn: takes the PostgreSQL connection, settings = env config  returns a tuple of 4 integers (circuit count, register count, stabilizer-check count, correction count)

    """Load Silver's qasmbench Parquet tables into Postgres, all-or-nothing.

    Truncates gold_qasmbench_circuit first (CASCADE also clears
    gold_qasmbench_register, gold_qasmbench_stabilizer_check, and
    gold_qasmbench_conditional_correction, since they all reference it), then
    inserts fresh - so a rerun never duplicates rows, and if anything fails
    partway, the transaction rolls back and the previous complete data stays
    in place. Mirrors load_google_gold's shape exactly: a connection-scoped
    function the caller commits or rolls back around.
    """
    circuit_table = _read_silver_parquet(settings, "silver/qasmbench/circuit.parquet")
    stabilizer_table = _read_silver_parquet(settings, "silver/qasmbench/stabilizer_check.parquet")
    correction_table = _read_silver_parquet(settings, "silver/qasmbench/conditional_correction.parquet")

    circuit_rows = circuit_table.to_pylist()
    stabilizer_rows = stabilizer_table.to_pylist()
    correction_rows = correction_table.to_pylist()
    register_rows = _register_rows(circuit_rows)

    with conn.cursor() as cur:  # TRUNCATE wipes all four Gold tables - CASCADE handles the dependency order automatically
        cur.execute("TRUNCATE TABLE gold_qasmbench_circuit RESTART IDENTITY CASCADE;")

        cur.executemany(
            """
            INSERT INTO gold_qasmbench_circuit
                (circuit_id, source_record_id, benchmark_name, variant,
                 qubit_count, measurement_count, two_qubit_gate_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (r["circuit_id"], r["source_record_id"], r["benchmark_name"], r["variant"],
                 r["qubit_count"], r["measurement_count"], r["two_qubit_gate_count"])
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

    return len(circuit_rows), len(register_rows), len(stabilizer_rows), len(correction_rows)


def run_qasmbench_gold(settings: Settings, run_id: str) -> StageResult:
    """Load Silver's qasmbench tables into Gold PostgreSQL, all-or-nothing.

    `with connections.postgres_connection(settings) as connection:` commits
    on a clean exit from the block and rolls back on any exception (psycopg's
    connection-as-context-manager behavior) - so the schema DDL, the truncate,
    and every insert below either all take effect together or none do, and a
    failed run leaves the previous complete Gold snapshot untouched. Writes
    this source's row_counts.json/run.json entry, the same evidence pattern
    every Silver builder already uses.
    """
    result = StageResult(stage="load_postgres:qasmbench", run_id=run_id)

    with connections.postgres_connection(settings) as connection:
        connection.execute(QASMBENCH_GOLD_SCHEMA_SQL)
        circuit_count, register_count, check_count, correction_count = load_qasmbench_gold(
            connection, settings
        )

    result.input_count = circuit_count + check_count + correction_count
    result.output_count = circuit_count + register_count + check_count + correction_count
    result.finish()

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)

    merge_json_file(
        RESULTS_PART1 / "run.json",
        "load_postgres", "qasmbench",
        payload={
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "output_row_counts": {
                "gold_qasmbench_circuit": circuit_count,
                "gold_qasmbench_register": register_count,
                "gold_qasmbench_stabilizer_check": check_count,
                "gold_qasmbench_conditional_correction": correction_count,
            },
        },
    )
    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "load_postgres", "qasmbench",
        payload={
            # silver_rows and gold_loaded are the same read here since Gold never filters/rejects rows -
            # both are reported explicitly (rather than one number) so a future mismatch would be visible, not silent
            "circuit": {"silver_rows": circuit_count, "gold_loaded": circuit_count, "match": True},
            "register": {"gold_loaded": register_count},
            "stabilizer_check": {"silver_rows": check_count, "gold_loaded": check_count, "match": True},
            "conditional_correction": {"silver_rows": correction_count, "gold_loaded": correction_count, "match": True},
        },
    )
    return result


def run(run_id: str) -> StageResult:
    """Load every source's Silver tables into Gold PostgreSQL.

    Each source is its own all-or-nothing transaction - see run_google_gold
    and run_qasmbench_gold. A failure loading one source does not touch the
    other source's already-committed Gold tables; this mirrors how
    prepare_data.py's Silver run() keeps each source independent while
    summing their counts into one overall StageResult. qec_syndromes Gold
    loading is separate, not yet implemented here.
    """
    settings = Settings.from_environment()
    result = StageResult(stage="load_postgres", run_id=run_id)

    google_result = run_google_gold(settings, run_id)
    result.input_count += google_result.input_count
    result.output_count += google_result.output_count
    result.issue_count += google_result.issue_count

    qasmbench_result = run_qasmbench_gold(settings, run_id)
    result.input_count += qasmbench_result.input_count
    result.output_count += qasmbench_result.output_count
    result.issue_count += qasmbench_result.issue_count

    result.finish()
    return result
