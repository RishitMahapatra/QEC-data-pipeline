"""Tests for the Bronze register_sources stage.

Two groups of tests:

- structural-check unit tests: exercise `_check_syndrome_structure`,
  `_check_google_structure`, and `_check_qasmbench_structure` directly against
  small in-memory zip files, both valid and deliberately broken. These need no
  platform at all.
- `run()` tests: exercise the full stage, including repeatability on
  unchanged input and that a bad archive actually stops the run. MinIO and the
  course manifest are faked via monkeypatching so these do not need the live
  Docker platform.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from quantum_lake_student.stages import register_sources as rs

SYNDROME_RATES = (
    "0.000010",
    "0.000050",
    "0.000100",
    "0.000500",
    "0.001000",
    "0.005000",
    "0.010000",
)
GOOGLE_EXPERIMENT_DIRS = (
    "surface_code_bX_d3_r25_center_3_5",
    "surface_code_bX_d3_r25_center_5_3",
    "surface_code_bX_d3_r25_center_5_7",
    "surface_code_bX_d3_r25_center_7_5",
    "surface_code_bX_d5_r25_center_5_5",
)

_FILENAMES = {
    "qec_syndromes": "syndromes_dataset.zip",
    "google_qec": "google-surface-code-curated.zip",
    "qasmbench": "qasmbench-qec.zip",
}


def _syndrome_zip_bytes(*, flaw: str | None = None) -> bytes:
    """Build a minimal but structurally-correct qec_syndromes zip, or one with one flaw."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for index, rate in enumerate(SYNDROME_RATES):
            if flaw == "missing_file" and index == 0:
                continue
            name = f"d-3_pfr-{rate}_nb-10M.csv"
            if flaw == "bad_filename" and index == 0:
                name = "unexpected_name.csv"
            header = rs.SYNDROME_CSV_HEADER
            if flaw == "bad_header" and index == 0:
                header = "label,syndromes,quantity"
            row = '0,"((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",100'
            zf.writestr(name, f"{header}\n{row}\n")
    return buffer.getvalue()


def _google_zip_bytes(*, flaw: str | None = None) -> bytes:
    """Build a minimal but structurally-correct google_qec zip, or one with one flaw."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        dirs = GOOGLE_EXPERIMENT_DIRS[:-1] if flaw == "missing_experiment" else GOOGLE_EXPERIMENT_DIRS
        for index, experiment_dir in enumerate(dirs):
            for name in rs.GOOGLE_REQUIRED_COMPANIONS:
                if flaw == "missing_companion" and index == 0 and name == "sweep.b8":
                    continue
                zf.writestr(f"{experiment_dir}/{name}", b"placeholder")
        if flaw == "unsafe_path":
            zf.writestr("../escape.txt", b"oops")
    return buffer.getvalue()


def _qasmbench_zip_bytes(*, flaw: str | None = None) -> bytes:
    """Build a minimal but structurally-correct qasmbench zip, or one with one flaw."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for index, circuit in enumerate(rs.QASMBENCH_CIRCUITS):
            for variant in (f"{circuit}.qasm", f"{circuit}_transpiled.qasm"):
                if flaw == "missing_variant" and index == 0 and variant.endswith("_transpiled.qasm"):
                    continue
                zf.writestr(f"small/{circuit}/{variant}", b"OPENQASM 2.0;\n")
    return buffer.getvalue()


_BUILDERS = {
    "qec_syndromes": _syndrome_zip_bytes,
    "google_qec": _google_zip_bytes,
    "qasmbench": _qasmbench_zip_bytes,
}


# ---------------------------------------------------------------------------
# Structural-check unit tests - plain zip bytes, no platform involved.
# ---------------------------------------------------------------------------


def test_is_unsafe_member_rejects_traversal_and_absolute_paths() -> None:
    assert rs.is_unsafe_member("../escape.txt")
    assert rs.is_unsafe_member("/etc/passwd")
    assert rs.is_unsafe_member("C:evil.txt")
    assert not rs.is_unsafe_member("small/qec_sm_n5/qec_sm_n5.qasm")


def test_check_syndrome_structure_accepts_valid_zip() -> None:
    data = _syndrome_zip_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        rs._check_syndrome_structure(zf, zf.namelist())  # must not raise


@pytest.mark.parametrize("flaw", ["missing_file", "bad_filename", "bad_header"])
def test_check_syndrome_structure_rejects_bad_data(flaw: str) -> None:
    data = _syndrome_zip_bytes(flaw=flaw)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with pytest.raises(rs.BronzeError):
            rs._check_syndrome_structure(zf, zf.namelist())


def test_check_google_structure_accepts_valid_zip() -> None:
    data = _google_zip_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        rs._check_google_structure(zf.namelist())  # must not raise


@pytest.mark.parametrize("flaw", ["missing_experiment", "missing_companion"])
def test_check_google_structure_rejects_bad_data(flaw: str) -> None:
    data = _google_zip_bytes(flaw=flaw)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with pytest.raises(rs.BronzeError):
            rs._check_google_structure(zf.namelist())


def test_check_qasmbench_structure_accepts_valid_zip() -> None:
    data = _qasmbench_zip_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        rs._check_qasmbench_structure(zf.namelist())  # must not raise


def test_check_qasmbench_structure_rejects_missing_variant() -> None:
    data = _qasmbench_zip_bytes(flaw="missing_variant")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with pytest.raises(rs.BronzeError):
            rs._check_qasmbench_structure(zf.namelist())


# ---------------------------------------------------------------------------
# Full run() tests - MinIO and the manifest are faked via monkeypatch, so
# these run without the live Docker platform.
# ---------------------------------------------------------------------------


def _stage_fake_release(
    tmp_path: Path, *, flaws: dict[str, str] | None = None
) -> tuple[Path, Path, dict[str, bytes]]:
    """Write placeholder Bronze files plus a matching manifest for one fake release.

    Returns (lake_root, manifest_path, zip_bytes_by_object_name). The
    placeholder files on disk only need to exist for `bronze_inventory`'s
    local-backend listing; the real zip bytes `run()` "downloads" are
    supplied separately via a monkeypatched `_read_object_bytes`, keyed by
    object name.
    """
    flaws = flaws or {}
    lake_root = tmp_path / "lake"
    manifest_objects = []
    zip_bytes_by_object_name: dict[str, bytes] = {}

    for source, builder in _BUILDERS.items():
        data = builder(flaw=flaws.get(source))
        filename = _FILENAMES[source]
        bronze_path = lake_root / "bronze" / f"source={source}" / filename
        bronze_path.parent.mkdir(parents=True, exist_ok=True)
        bronze_path.write_bytes(b"")

        object_name = f"bronze/source={source}/{filename}"
        zip_bytes_by_object_name[object_name] = data
        manifest_objects.append(
            {
                "source": source,
                "path": f"raw/source={source}/{filename}",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "mandatory": True,
            }
        )

    manifest_path = tmp_path / "bundle-manifest.json"
    manifest_path.write_text(json.dumps({"objects": manifest_objects}))

    return lake_root, manifest_path, zip_bytes_by_object_name


def _patch_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lake_root: Path,
    manifest_path: Path,
    zip_bytes_by_object_name: dict[str, bytes],
    results_root: Path,
) -> None:
    monkeypatch.setenv("LAKE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LAKE_ROOT", str(lake_root))
    monkeypatch.setattr(rs, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(rs, "RESULTS_PART1", results_root)

    def _fake_read_object_bytes(settings, object_name):  # noqa: ARG001 - matches real signature
        return zip_bytes_by_object_name[object_name]

    monkeypatch.setattr(rs, "_read_object_bytes", _fake_read_object_bytes)


def test_run_succeeds_and_is_repeatable_on_unchanged_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake_root, manifest_path, zip_bytes = _stage_fake_release(tmp_path)
    results_root = tmp_path / "results" / "part1"
    _patch_platform(
        monkeypatch,
        lake_root=lake_root,
        manifest_path=manifest_path,
        zip_bytes_by_object_name=zip_bytes,
        results_root=results_root,
    )

    first = rs.run("test-run-1")
    first_rows = pq.read_table(results_root / "source_registration.parquet").to_pylist()
    first_run_json = json.loads((results_root / "run.json").read_text())

    second = rs.run("test-run-2")
    second_rows = pq.read_table(results_root / "source_registration.parquet").to_pylist()
    second_run_json = json.loads((results_root / "run.json").read_text())

    assert first.output_count == second.output_count == 3
    assert first.issue_count == second.issue_count == 0
    # Same input -> identical rows, and no duplicates accumulate across runs.
    assert first_rows == second_rows
    assert len(second_rows) == 3
    assert (
        first_run_json["register_sources"]["input_hashes"]
        == second_run_json["register_sources"]["input_hashes"]
    )


def test_run_raises_on_unsafe_archive_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake_root, manifest_path, zip_bytes = _stage_fake_release(
        tmp_path, flaws={"google_qec": "unsafe_path"}
    )
    results_root = tmp_path / "results" / "part1"
    _patch_platform(
        monkeypatch,
        lake_root=lake_root,
        manifest_path=manifest_path,
        zip_bytes_by_object_name=zip_bytes,
        results_root=results_root,
    )

    with pytest.raises(rs.BronzeError):
        rs.run("test-run-unsafe")


@pytest.mark.parametrize(
    "source,flaw",
    [
        ("qec_syndromes", "missing_file"),
        ("google_qec", "missing_companion"),
        ("qasmbench", "missing_variant"),
    ],
)
def test_run_raises_on_missing_required_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, flaw: str
) -> None:
    lake_root, manifest_path, zip_bytes = _stage_fake_release(tmp_path, flaws={source: flaw})
    results_root = tmp_path / "results" / "part1"
    _patch_platform(
        monkeypatch,
        lake_root=lake_root,
        manifest_path=manifest_path,
        zip_bytes_by_object_name=zip_bytes,
        results_root=results_root,
    )

    with pytest.raises(rs.BronzeError):
        rs.run("test-run-missing")
