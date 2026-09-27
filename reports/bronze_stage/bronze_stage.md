# Bronze stage

--Should we also answer the questions here?--

This section explains the main decisions taken in the Bronze stage and why
they were taken.

Code: [register_sources.py](../../starter/src/quantum_lake_student/stages/register_sources.py) ·
Tests: [test_register_sources.py](../../starter/tests/test_register_sources.py)

## Input and output

- **Input:** the three mandatory source archives, which `make bootstrap`
  copies unchanged into MinIO under `bronze/`:
  - `qec_syndromes`: `syndromes_dataset.zip`
  - `google_qec`: `google-surface-code-curated.zip`
  - `qasmbench`: `qasmbench-qec.zip`

  The course manifest `bundle-manifest.json` gives the expected SHA-256 and
  size of each archive.
- **Output:**
  - No data is transformed.
  - The stage records what was verified in `results/part1/run.json` (run id,
    code revision, timestamps, input hashes, unexpected objects).
  - It also records counts in `results/part1/row_counts.json` (read, accepted
    and rejected objects, plus the size and member count of each archive).

## Main decisions

**1. Bronze is read-only.** The stage only reads the archives into memory.
It never extracts, edits or rewrites them. Every later stage reads these
original bytes, and every Silver trace row points back to them through the
archive name and its hash. The spec requires this: "preserve original bytes,
and never correct data in place".

**2. Integrity check against the manifest.** For each mandatory archive the
stage computes the SHA-256 and the size in bytes and compares both with the
manifest. Any mismatch stops the run. This proves that the files are exactly
the ones the course supplied.

**3. Archive paths are checked for safety before anything else.** Any zip
member with an absolute path (`/…`), a `..` segment or a drive letter (`C:`)
is rejected, because such a path could write outside the target folder if the
archive were ever extracted.

**4. The structure is checked, not the content.** Each source gets a check
that looks only at file names, file counts and the folder layout. Parsing
the contents is the job of Silver.

| Source | Structural check |
| --- | --- |
| `qec_syndromes` | Exactly 7 CSV files named `d-3_pfr-<rate>_nb-10M.csv`, each with the header `labels,syndromes,quantity` |
| `google_qec` | Exactly 4 distance-3 and 1 distance-5 experiment folders with the documented name pattern, each with all 15 companion files (`properties.yml`, `b8` and `01` files, circuits, layout, error models) |
| `qasmbench` | Exactly the 3 documented circuits, each with its source and `_transpiled` `.qasm` file, and no extra circuit folders |

**5. Missing or unexpected objects are handled differently.**
- A missing mandatory archive, a hash or size mismatch, an unsafe path or a
  structural error **stops the run** with a `BronzeError`. If the input is
  wrong, nothing downstream can be trusted, so bad data must not flow on
  silently.
- An extra object in Bronze that the manifest does not list does **not** stop
  the run, because it does not affect the mandatory archives. It is reported
  as a warning and counted as an issue, and the list of such objects is saved
  in `run.json`.

**6. Reruns are safe.** Nothing is ever written to Bronze. The result files
are merged under the `register_sources` key, so a second run replaces this
stage's entry and creates no duplicates.

## Requirements coverage

| Requirement | How it is met |
| --- | --- |
| Keep the supplied bytes unchanged | Decision 1 |
| Verify checksums from the release description | Decision 2 |
| List archive members safely | Decision 3 |
| Confirm each source has its documented files | Decision 4 |
| Report missing or unexpected source objects | Decision 5 |
| A second run creates no duplicate records | Decision 6 |

All 24 tests pass: `cd starter && uv run pytest tests/test_register_sources.py`

**Result on the supplied data:** all 3 objects are read and accepted, and 0
are rejected. The member counts are 8 for `qec_syndromes`, 76 for `google_qec`
and 19 for `qasmbench`.

## Limitations

- The structural checks are tied to this specific course release (for example
  exactly 7 CSVs and 5 Google experiments). A new release with more files
  would need the expected counts updated.
