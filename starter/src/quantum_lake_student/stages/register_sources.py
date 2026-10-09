
# ─────────────────────────────────────────────────────────────────────────
# OVERVIEW
#
# This file is the Bronze-stage entry point for the pipeline. Its job is to prove those three files are exactly what the
# course says they should be, before anything downstream (Silver parsing, Gold loading, ML export) is allowed to trust them.
#
# Concretely, for each of the 3 mandatory sources (qec_syndromes, google_qec,
# qasmbench), this file:
#   1. Downloads the object from Bronze and checks its hash + size against
#      the course-supplied manifest (proves the bytes are unmodified).
#   2. Opens it as a zip and rejects any unsafe internal path (../, absolute
#      paths, drive letters) before anything is ever extracted.
#   3. Runs a source-specific structural check (right file count, right
#      names, right folder layout) WITHOUT parsing any actual file content —
#      that parsing belongs to Silver, not here.
#   4. Records what happened into run.json / row_counts.json, so every
#      later pipeline run has a paper trail of exactly what was verified,
#      when, and against which hashes.
#
# Everything here is read-only against Bronze, this file never edits,
# extracts, or rewrites the source zips. If any check fails, it raises
# BronzeError, which is meant to stop the whole run rather than let bad
# data flow forward without a clear error.
# ─────────────────────────────────────────────────────────────────────────


import io #treats the raw bytes file as a normal file so that zipfile can open them without accessing the disk
import zipfile # for reading the zipfile directly from memory
import json #for handling the JSON files
import re #request -> for pattern matching for file names and folders
from pathlib import Path 

from quantum_lake_student.models import RESULTS_PART1, StageResult, code_revision, merge_json_file

MANIFEST_PATH = Path("/course-data/metadata/bundle-manifest.json") #the path is where the assignment requires the hash files to be located 


def _load_manifest() -> dict:
    """Read the course-supplied manifest describing every mandatory source object."""
    return json.loads(MANIFEST_PATH.read_text()) #  parsing the manifest JSON into a Python dict


def _bronze_key(manifest_path: str) -> str:
    """Map a manifest path like 'raw/source=X/f.zip' to its Bronze key 'bronze/source=X/f.zip'."""
    return manifest_path.replace("raw/", "bronze/", 1)

 
import hashlib #hashlib uses the SHA-256 encryption algo. to verify file integrity

from quantum_lake_student.connections import Settings, minio_client, bronze_inventory # env config + MinIO client factory + inventory listing


class BronzeError(RuntimeError):
    """A Bronze object failed verification. The caller must stop the run."""
#a custom exception type so callers can tell "bad data" apart from any other error
       
def _read_object_bytes(settings: Settings, object_name: str) -> bytes:
    """Fetch one Bronze object's raw bytes from MinIO, fully unread into memory."""
    client = minio_client(settings) #getting the minio client using the env settings
    resp = client.get_object(settings.s3_bucket, object_name) # start streaming with the object's bytes
    try:
        return resp.read() # pull the entire object into memory as bytes
    finally:
        resp.close() #closing the HTTP connnection
        resp.release_conn() #returning connection to the HTTP pool to avoid dataleakage



def _bronze_object_name(manifest_path: str, inventory: dict[str, int]) -> str | None:
    """Map a manifest path to its Bronze key, but only if that key actually exists in Bronze."""
    candidate = _bronze_key(manifest_path)
    return candidate if candidate in inventory else None
        

def _verify_object(settings: Settings, manifest_entry: dict, inventory: dict[str, int]) -> tuple[str, bytes]:
    """Fetch one object and confirm its hash and size match the manifest exactly."""
    object_name = _bronze_object_name(manifest_entry["path"], inventory)
    if object_name is None:
        raise BronzeError(f"missing mandatory source object: {manifest_entry['path']}")

    data = _read_object_bytes(settings, object_name) 

    actual_hash = hashlib.sha256(data).hexdigest() #computing the hash download
    if actual_hash != manifest_entry["sha256"]: #compare against the expected hash
        raise BronzeError( #raises error in case of hash mismatch
            f"{object_name}: hash mismatch — expected {manifest_entry['sha256']}, "
            f"got {actual_hash}"
        )

    if len(data) != manifest_entry["bytes"]: #checking if the expected number of bytes matches the current extracted bytes. If not, then raise the BronzeError. 
        raise BronzeError(
            f"{object_name}: size mismatch — expected {manifest_entry['bytes']} bytes, "
            f"got {len(data)}"
        )

    return object_name, data

def is_unsafe_member(name: str) -> bool: #The main purpose of this function is to handle the file paths
    """True if a zip member path could escape the archive's extraction directory."""
    normalized = name.replace("\\", "/") # treating the Windows-style backslashes the same as forward slashes
    parts = normalized.split("/") #breaking the paths into individual segments
    return normalized.startswith("/") or ".." in parts or ":" in parts[0] # unsafe if: it's an absolute path, contains a ".." traversal segment, or starts with a drive letter


def _check_archive_safety(object_name: str, data: bytes) -> list[str]:
    """Open a Bronze zip and return the list of its member names, after confirming none are unsafe."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()

    unsafe = [m for m in members if is_unsafe_member(m)]
    if unsafe:
        raise BronzeError(f"{object_name}: unsafe archive member(s): {unsafe}")

    return members  # return the validated member list for reuse by other checks


SYNDROME_CSV_PATTERN = re.compile(r"^d-3_pfr-\d+\.\d{6}_nb-10M\.csv$")  #the documented syndrome filename shape
SYNDROME_CSV_HEADER = "labels,syndromes,quantity" #The exact expected csv header line

#This function is to check whether the file name matches the specifications of the dataset 

#Starting the verification with the Syndrome dataset 
def _check_syndrome_structure(data: bytes, members: list[str]) -> None:
    """Confirm the qec_syndromes archive has exactly 7 correctly-named CSVs with the right header."""
    csv_members = sorted(m for m in members if m.endswith(".csv")) #extracting all the csv files inside the mentioned folder

    if len(csv_members) != 7: #since there are 7 datasets for syndromes, the count must match
        raise BronzeError( #else raise the BronzeError
            f"expected 7 qec_syndromes CSV files, found {len(csv_members)}: {csv_members}"
        )

    with zipfile.ZipFile(io.BytesIO(data)) as zf: #checking if the file name matches the documented pattern
        for member in csv_members:
            if not SYNDROME_CSV_PATTERN.match(member):
                raise BronzeError(
                    f"qec_syndromes file name does not match the documented "
                    f"pattern: {member}"
                )
            header = zf.read(member).split(b"\n", 1)[0].decode("utf-8").strip() #reading the first line
            if header != SYNDROME_CSV_HEADER: #confirms if the header matches exactly
                raise BronzeError(
                    f"{member}: header {header!r} does not match the "
                    f"documented header {SYNDROME_CSV_HEADER!r}"
                )

#verification of the files in the google QEC shot datasets
GOOGLE_DIR_PATTERN = re.compile(r"^surface_code_bX_d(\d+)_r\d+_center_\d+_\d+$") #capturing the d value (distance digit)
GOOGLE_EXPECTED_D3_COUNT = 4 #there are exactly 4 distance experiments mentioned 
GOOGLE_EXPECTED_D5_COUNT = 1 # exactly 1 distance-5 experiment folder expected

GOOGLE_REQUIRED_COMPANIONS = (  # listing every file that must exist inside each experiment folder
    "properties.yml",
    "measurements.b8",
    "sweep.b8",
    "detection_events.b8",
    "obs_flips_actual.01",
    "obs_flips_predicted_by_belief_matching.01",
    "obs_flips_predicted_by_correlated_matching.01",
    "obs_flips_predicted_by_pymatching.01",
    "obs_flips_predicted_by_tensor_network_contraction.01",
    "circuit_ideal.stim",
    "circuit_noisy.stim",
    "circuit_detector_error_model.dem",
    "layout.svg",
    "pij_from_even_for_odd.dem",
    "pij_from_odd_for_even.dem",
)


def _check_google_structure(members: list[str]) -> None:
    """Confirm google_qec has exactly 4 distance-3 + 1 distance-5 experiment
    directories, each with every required companion file."""
    experiment_dirs = sorted({m.split("/", 1)[0] for m in members if "/" in m})

    d3_dirs = [] #this will hold all the files with distance d = 3
    d5_dirs = [] #this will hold all the files with distance d = 5
    for name in experiment_dirs:
        match = GOOGLE_DIR_PATTERN.match(name) #extracting the digit from the folder name
        if not match:
            raise BronzeError(f"google_qec directory name is not in the documented format: {name}")
        distance = match.group(1) # pull out the captured distance number
        if distance == "3": #if the distance value is 3
            d3_dirs.append(name)  #add it to the list of the d3_dirs
        elif distance == "5": #else if distance value is 5
            d5_dirs.append(name) #add it to the list of the d5_dirs
        else: #if none of the extracted digit matches to 3 or 5 -> raise the BronzeError 
            raise BronzeError(f"google_qec directory has an unexpected distance value: {name}")

    if len(d3_dirs) != GOOGLE_EXPECTED_D3_COUNT or len(d5_dirs) != GOOGLE_EXPECTED_D5_COUNT: #if none of the file count matches the expected count (for both d3 or d5) -> raise the BronzeError
        raise BronzeError(
            f"expected {GOOGLE_EXPECTED_D3_COUNT} distance-3 and {GOOGLE_EXPECTED_D5_COUNT} "
            f"distance-5 directories, found {len(d3_dirs)} distance-3 and {len(d5_dirs)} distance-5"
        )

    member_set = set(members) # Convert to a set for fast "is this path present" checks
    for experiment_dir in experiment_dirs: # find any required companion file that's absent from this folder
        missing = [
            name for name in GOOGLE_REQUIRED_COMPANIONS
            if f"{experiment_dir}/{name}" not in member_set
        ]
        if missing: #will raise the BronzeError if any companion file is missing 
            raise BronzeError(
                f"google_qec experiment {experiment_dir!r} is missing required companion file(s): {missing}"
            )

#Now building the QASM structural check 
QASMBENCH_CIRCUITS = ("error_correctiond3_n5", "qec_en_n5", "qec_sm_n5") #contains the documented circuit names


def _check_qasmbench_structure(members: list[str]) -> None:
    """Confirm qasmbench has exactly the 3 documented circuits, each with
    both its source and transpiled QASM variant, and no extra circuit directories."""
    member_set = set(members) #using set for fast lookup 

    for circuit in QASMBENCH_CIRCUITS:
        for variant in (f"{circuit}.qasm", f"{circuit}_transpiled.qasm"):
            path = f"small/{circuit}/{variant}"
            if path not in member_set:
                raise BronzeError(f"qasmbench is missing required circuit file: {path}")

    actual_circuit_dirs = sorted({ # list every actual circuit-folder name found in the archive
        m.split("/")[1] for m in members
        if m.startswith("small/") and m.count("/") >= 2
    })
    expected_circuit_dirs = sorted(QASMBENCH_CIRCUITS) #the exact set we expect to see
    if actual_circuit_dirs != expected_circuit_dirs: #comparing the 2 sets 
        extra = set(actual_circuit_dirs) - set(expected_circuit_dirs) 
        missing = set(expected_circuit_dirs) - set(actual_circuit_dirs)
        raise BronzeError(
            f"qasmbench circuit directories do not match exactly — "
            f"extra: {sorted(extra) or 'none'}, missing: {sorted(missing) or 'none'}"
        )
        
# run function is the main entry point of the pipeline where the code written will tie every function together
def run(run_id: str) -> StageResult:
    """Verify and register the three mandatory Bronze source objects."""
    settings = Settings.from_environment() #loading connection config from environment variables 
    result = StageResult(stage="register_sources", run_id=run_id) # start tracking this run's outcome
    inventory = dict(bronze_inventory(settings))

    manifest = _load_manifest() #read the expected-objects manifest
    mandatory_objects = [obj for obj in manifest["objects"] if obj["mandatory"]] # keep only the required ones
    result.input_count = len(mandatory_objects)  # record how many objects we expect to process

    registrations: list[dict] = [] #collecting one summary dict per verified object
    expected_names: set[str] = set()  # every Bronze key the manifest says should exist

    for obj in mandatory_objects:
        expected_names.add(_bronze_key(obj["path"]))
        object_name, data = _verify_object(settings, obj, inventory)
        members = _check_archive_safety(object_name, data)

        #routing to the correct file based on the source name 
        if obj["source"] == "qec_syndromes":
            _check_syndrome_structure(data, members)
        elif obj["source"] == "google_qec":
            _check_google_structure(members)
        elif obj["source"] == "qasmbench":
            _check_qasmbench_structure(members)

        registrations.append({ # record this object's verification summary
            "source_name": obj["source"],
            "bronze_object": object_name,
            "input_sha256": obj["sha256"],
            "size_bytes": len(data),
            "member_count": len(members),
        })
    unexpected = sorted(set(inventory) - expected_names)
    if unexpected:
        print(f"WARNING: unexpected Bronze objects present: {unexpected}")

    result.output_count = len(registrations)  # how many objects were successfully verified
    result.issue_count = len(unexpected)  # unexpected objects counted as issues, not silently ignored
    result.finish()  # mark the finish timestamp on the result

    RESULTS_PART1.mkdir(parents=True, exist_ok=True)  # ensure the results folder exists before writing to it

    merge_json_file(
        RESULTS_PART1 / "run.json",
        "register_sources",
        payload={ # this stage's contribution to the shared run record
            "run_id": run_id,
            "code_revision": code_revision(),
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "input_count": result.input_count,
            "output_count": result.output_count,
            "issue_count": result.issue_count,
            "input_hashes": {
                r["source_name"]: r["input_sha256"] for r in registrations
            },
            "unexpected_bronze_objects": unexpected,  
        },
    )

    merge_json_file(
        RESULTS_PART1 / "row_counts.json",
        "register_sources",
        payload={ # this stage's contribution to the shared row-count record
            "read": result.input_count,
            "accepted": result.output_count,
            "rejected": result.issue_count,
            "objects": {
                r["source_name"]: {
                    "member_count": r["member_count"],
                    "size_bytes": r["size_bytes"],
                }
                for r in registrations
            },
        },
    )

    return result