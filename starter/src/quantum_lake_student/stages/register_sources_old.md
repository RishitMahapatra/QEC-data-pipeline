# Documentation: `register_sources.py`

This document provides a detailed explanation of each function in `register_sources.py`, along with its expected behavior and output.

---

## Shared Helpers

### `is_safe_member_path(name: str) -> bool`
**Description:** Checks if a file path inside an archive (like a ZIP file) is safe to extract. It prevents directory traversal attacks by rejecting absolute paths, Windows drive letters, and paths containing `..`.
**Expected Output:** 
- Returns `True` if the path is safe (e.g., `folder/file.txt`).
- Returns `False` if unsafe (e.g., `../file.txt`, `/etc/passwd`, `C:\Windows`).

### `_finding(...) -> QualityFinding`
**Description:** A private helper function used to standardize the creation of `QualityFinding` objects. These objects represent issues found during validation (e.g., file missing, hash mismatch).
**Expected Output:** Returns a `QualityFinding` object containing the rule ID, severity, source, location, message, and observed values.

---

## Release Manifest

### `load_release_manifest(metadata_dir: Path) -> list[ManifestObject]`
**Description:** Reads the `bundle-manifest.json` file from the metadata directory and parses it into a list of `ManifestObject` data classes. This manifest acts as the ground truth for what files should exist, their expected sizes, and SHA-256 hashes.
**Expected Output:** A list of `ManifestObject` instances representing the declared dataset files.

---

## Hashing, Archive Integrity, and Safe Listing

### `sha256_of_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str`
**Description:** Calculates the SHA-256 hash of a given file. It reads the file in chunks (default 1MB) to prevent memory overload when hashing massive dataset files.
**Expected Output:** A lowercase hexadecimal string representing the file's SHA-256 hash (e.g., `"a1b2c3d4..."`).

### `list_archive_members(path: Path) -> list[dict[str, Any]]`
**Description:** Inspects a ZIP archive and lists all its contents without actually extracting any bytes.
**Expected Output:** A list of dictionaries, where each dictionary contains:
- `name`: The file/folder path inside the ZIP.
- `bytes`: The uncompressed size of the file.
- `is_dir`: A boolean indicating if it is a directory.

### `check_archive_integrity(path: Path) -> str | None`
**Description:** Uses Python's built-in `zipfile.testzip()` method to run a CRC (Cyclic Redundancy Check) on every member inside the ZIP archive to ensure the ZIP is not corrupted.
**Expected Output:** 
- Returns `None` if the archive is perfectly intact.
- Returns the string name of the first corrupt file if an error is found.

---

## Per-Object Verification

### `BronzeVerificationResult.merge(self, other: "BronzeVerificationResult") -> None`
**Description:** Merges the findings and records of another verification result into the current one. Useful for combining results from multiple verified files into a single master summary.
**Expected Output:** Modifies the object in-place (returns `None`).

### `BronzeVerificationResult.summary(self) -> dict[str, Any]`
**Description:** Generates a high-level summary dictionary of the verification run, counting declared objects, verified objects, and grouping issues by severity and rule type.
**Expected Output:** A dictionary containing stats like `{"objects_declared": 3, "objects_verified": 3, "issue_count": 0, ...}`.

### `verify_bronze_object(local_path: Path, manifest_entry: ManifestObject) -> BronzeVerificationResult`
**Description:** The core validation engine for a single file. It compares a local file against its expected `ManifestObject` entry. It checks:
1. File existence.
2. Exact byte size match.
3. Exact SHA-256 hash match.
4. Archive corruption.
5. Safe archive member paths.
If no strict errors occur, it generates a `BronzeObjectRecord`.
**Expected Output:** A `BronzeVerificationResult` object populated with any detected `QualityFinding` errors/warnings and a `BronzeObjectRecord` if verification succeeded.

---

## Missing / Unexpected Object Inventory

### `discover_actual_objects(raw_root: Path) -> dict[str, Path]`
**Description:** Scans the physical `raw/` directory to see what files actually exist on the disk, formatting their paths to match the manifest format (`raw/<...>`).
**Expected Output:** A dictionary mapping relative paths to absolute `Path` objects (e.g., `{"raw/source/file.zip": PosixPath('/.../file.zip')}`).

### `check_unexpected_objects(manifest: list[ManifestObject], actual: dict[str, Path]) -> list[QualityFinding]`
**Description:** Compares the actual files on disk against the expected files in the manifest. Identifies leftover files, OS junk (like `.DS_Store`), or manually added files that shouldn't be there.
**Expected Output:** A list of `QualityFinding` objects with severity `WARNING` for any undocumented files.

---

## Orchestration

### `register_bronze_sources(bundle_root: Path) -> BronzeVerificationResult`
**Description:** The main driver for the verification process. It loads the manifest, discovers actual files, checks for unexpected files, and iterates through the manifest to verify each file individually.
**Expected Output:** A master `BronzeVerificationResult` containing the combined records and findings of the entire dataset.

### `_default_bundle_root() -> Path`
**Description:** Attempts to automatically find the dataset bundle directory by checking environment variables (`QEC_BUNDLE_ROOT`), Docker mount paths (`/course-data`), or traversing relative to the script location.
**Expected Output:** A `Path` object pointing to the dataset root. Raises `FileNotFoundError` if unsuccessful.

---

## Output Generation

### `_findings_to_rows(findings: list[QualityFinding]) -> list[dict[str, Any]]`
**Description:** Converts internal `QualityFinding` objects into flat dictionaries, preparing them to be saved to tabular formats like Parquet or CSV.
**Expected Output:** A list of dictionaries formatted for tabular export.

### `write_data_issues(rows: list[dict[str, Any]], output_path: Path) -> Path`
**Description:** Saves the verification issues to disk as a Parquet file (`data_issues.parquet`). If `pandas`/`pyarrow` are missing, it gracefully falls back to saving a CSV file.
**Expected Output:** Returns the `Path` to the saved output file.

### `write_bronze_inventory(result: BronzeVerificationResult, output_path: Path) -> Path`
**Description:** Saves the successful verification records and high-level summary to disk as a structured JSON file (`bronze_inventory.json`).
**Expected Output:** Returns the `Path` to the saved JSON file.

### `_execute(bundle_root: Path, out_dir: Path) -> tuple[BronzeVerificationResult, Path, Path]`
**Description:** Runs the orchestration logic and then writes the output files to the designated results directory.
**Expected Output:** A tuple containing:
1. The `BronzeVerificationResult`.
2. The path to the saved issues file.
3. The path to the saved inventory file.

### `run(run_id: str, *, bundle_root: Path | None = None, results_dir: Path | None = None) -> StageResult`
**Description:** The formal entry point used by the broader data pipeline framework. It triggers `_execute` and packages the final result into a standard `StageResult` object.
**Expected Output:** A `StageResult` indicating whether the stage succeeded or failed, along with execution counts.

### `_print_summary(result: BronzeVerificationResult) -> None`
**Description:** Uses the `rich` library to print a nicely formatted terminal table summarizing the verification run, hashes, and any encountered issues.
**Expected Output:** Terminal output (returns `None`).

### `build_arg_parser() -> argparse.ArgumentParser`
**Description:** Configures command-line arguments (e.g., `--bundle-root`, `--results-dir`) so the script can be run flexibly from the terminal.
**Expected Output:** An `ArgumentParser` object.

### `main(argv: list[str] | None = None) -> int`
**Description:** The CLI entry point. Parses arguments, triggers the verification logic, prints the summary, and returns an exit code.
**Expected Output:** 
- Returns `0` if verification passed or only had warnings.
- Returns `1` if there were any severe errors (e.g., missing files, hash mismatches).
