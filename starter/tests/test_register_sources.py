"""Tests for the Bronze register_sources stage.

Three groups of tests:

- archive-safety and structural-check unit tests: exercise
  `is_unsafe_member`, `_check_syndrome_structure`, `_check_google_structure`,
  and `_check_qasmbench_structure` directly against small in-memory zip
  files, both valid and deliberately broken. These need no platform at all.
- `_verify_object` unit tests: exercise hash/size verification against a
  fake manifest entry, both matching and mismatched, plus the case where
  the object is missing from the Bronze inventory entirely.
- `run()` tests: exercise the full stage end to end, including
  repeatability on unchanged input and that a bad archive actually stops
  the run. MinIO, `bronze_inventory`, and the course manifest are faked via
  monkeypatching so these do not need the live Docker platform.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

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
        dirs = GOOGLE_EXPERIMENT_DIRS
        if flaw == "wrong_distance_split":
            # swap the one d5 dir for a second copy of a d3 dir name — total
            # count stays 5, but the 4+1 split is violated
            dirs = GOOGLE_EXPERIMENT_DIRS[:-1] + ("surface_code_bX_d3_r25_center_9_9",)
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
        circuits = rs.QASMBENCH_CIRCUITS
        if flaw == "extra_circuit":
            circuits = circuits + ("unexpected_circuit_n5",)
        for index, circuit in enumerate(circuits):
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
# Archive-safety and structural-check unit tests - plain zip bytes, no
# platform involved.
# ---------------------------------------------------------------------------


def test_is_unsafe_member_rejects_traversal_absolute_and_drive_paths() -> None:
    assert rs.is_unsafe_member("../escape.txt")
    assert rs.is_unsafe_member("/etc/passwd")
    assert rs.is_unsafe_member("C:/Windows/System32")
    assert not rs.is_unsafe_member("small/qec_sm_n5/qec_sm_n5.qasm")


def test_check_archive_safety_rejects_unsafe_member() -> None:
    data = _google_zip_bytes(flaw="unsafe_path")
    with pytest.raises(rs.BronzeError):
        rs._check_archive_safety("bronze/source=google_qec/fake.zip", data)


def test_check_syndrome_structure_accepts_valid_zip() -> None:
    data = _syndrome_zip_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
    rs._check_syndrome_structure(data, members)  # must not raise


@pytest.mark.parametrize("flaw", ["missing_file", "bad_filename", "bad_header"])
def test_check_syndrome_structure_rejects_bad_data(flaw: str) -> None:
    data = _syndrome_zip_bytes(flaw=flaw)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
    with pytest.raises(rs.BronzeError):
        rs._check_syndrome_structure(data, members)


def test_check_google_structure_accepts_valid_zip() -> None:
    data = _google_zip_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
    rs._check_google_structure(members)  # must not raise


@pytest.mark.parametrize(
    "flaw", ["missing_companion", "wrong_distance_split"]
)
def test_check_google_structure_rejects_bad_data(flaw: str) -> None:
    data = _google_zip_bytes(flaw=flaw)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
    with pytest.raises(rs.BronzeError):
        rs._check_google_structure(members)


def test_check_qasmbench_structure_accepts_valid_zip() -> None:
    data = _qasmbench_zip_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
    rs._check_qasmbench_structure(members)  # must not raise


@pytest.mark.parametrize("flaw", ["missing_variant", "extra_circuit"])
def test_check_qasmbench_structure_rejects_bad_data(flaw: str) -> None:
    data = _qasmbench_zip_bytes(flaw=flaw)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
    with pytest.raises(rs.BronzeError):
        rs._check_qasmbench_structure(members)


# ---------------------------------------------------------------------------
# _verify_object unit tests - hash/size checking against a fake manifest
# entry, with `_read_object_bytes` monkeypatched so no MinIO call happens.
# Each entry's path is converted to its Bronze key and placed in a fake
# `inventory` dict, since _verify_object now confirms the object is present
# in Bronze before ever trying to fetch it.
# ---------------------------------------------------------------------------


def test_verify_object_accepts_matching_hash_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"some bronze object bytes"
    entry = {
        "source": "qasmbench",
        "path": "raw/source=qasmbench/qasmbench-qec.zip",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mandatory": True,
    }
    monkeypatch.setattr(rs, "_read_object_bytes", lambda settings, name: data)
    inventory = {entry["path"].replace("raw/", "bronze/", 1): 0}

    object_name, returned_data = rs._verify_object(
        settings=None, manifest_entry=entry, inventory=inventory
    )

    assert object_name == "bronze/source=qasmbench/qasmbench-qec.zip"
    assert returned_data == data


def test_verify_object_rejects_hash_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = {
        "source": "qasmbench",
        "path": "raw/source=qasmbench/qasmbench-qec.zip",
        "bytes": 5,
        "sha256": "0" * 64,
        "mandatory": True,
    }
    monkeypatch.setattr(rs, "_read_object_bytes", lambda settings, name: b"wrong")
    inventory = {entry["path"].replace("raw/", "bronze/", 1): 0}

    with pytest.raises(rs.BronzeError):
        rs._verify_object(settings=None, manifest_entry=entry, inventory=inventory)


def test_verify_object_rejects_size_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"actual bytes"
    entry = {
        "source": "qasmbench",
        "path": "raw/source=qasmbench/qasmbench-qec.zip",
        "bytes": len(data) + 1,  # deliberately wrong
        "sha256": hashlib.sha256(data).hexdigest(),
        "mandatory": True,
    }
    monkeypatch.setattr(rs, "_read_object_bytes", lambda settings, name: data)
    inventory = {entry["path"].replace("raw/", "bronze/", 1): 0}

    with pytest.raises(rs.BronzeError):
        rs._verify_object(settings=None, manifest_entry=entry, inventory=inventory)


def test_verify_object_rejects_object_missing_from_inventory() -> None:
    """A manifest entry whose Bronze key isn't actually present must fail
    clearly, before any attempt to fetch it from MinIO."""
    entry = {
        "source": "qasmbench",
        "path": "raw/source=qasmbench/qasmbench-qec.zip",
        "bytes": 5,
        "sha256": "0" * 64,
        "mandatory": True,
    }
    with pytest.raises(rs.BronzeError):
        rs._verify_object(settings=None, manifest_entry=entry, inventory={})


# ---------------------------------------------------------------------------
# Full run() tests - MinIO, bronze_inventory, and the manifest are faked via
# monkeypatch, so these run without the live Docker platform.
# ---------------------------------------------------------------------------


def _stage_fake_manifest(
    tmp_path: Path, *, flaws: dict[str, str] | None = None
) -> tuple[Path, dict[str, bytes]]:
    """Build a fake manifest plus matching zip bytes for one fake release.

    Returns (manifest_path, zip_bytes_by_object_name). The zip bytes are
    supplied to `run()` via a monkeypatched `_read_object_bytes`, keyed by
    the resolved Bronze object name.
    """
    flaws = flaws or {}
    manifest_objects = []
    zip_bytes_by_object_name: dict[str, bytes] = {}

    for source, builder in _BUILDERS.items():
        data = builder(flaw=flaws.get(source))
        filename = _FILENAMES[source]
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

    return manifest_path, zip_bytes_by_object_name


def _patch_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_path: Path,
    zip_bytes_by_object_name: dict[str, bytes],
    results_root: Path,
) -> None:
    monkeypatch.setattr(rs, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(rs, "RESULTS_PART1", results_root)

    def _fake_read_object_bytes(settings, object_name):  # noqa: ARG001 - matches real signature
        return zip_bytes_by_object_name[object_name]

    monkeypatch.setattr(rs, "_read_object_bytes", _fake_read_object_bytes)

    # run() now calls bronze_inventory(settings) directly to cross-check
    # each manifest object's presence and to detect unexpected extras, so
    # this must return exactly the object names this fake release provides.
    def _fake_bronze_inventory(settings):  # noqa: ARG001 - matches real signature
        return [(name, 0) for name in zip_bytes_by_object_name]

    monkeypatch.setattr(rs, "bronze_inventory", _fake_bronze_inventory)


def test_run_succeeds_and_is_repeatable_on_unchanged_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, zip_bytes = _stage_fake_manifest(tmp_path)
    results_root = tmp_path / "results" / "part1"
    _patch_platform(
        monkeypatch,
        manifest_path=manifest_path,
        zip_bytes_by_object_name=zip_bytes,
        results_root=results_root,
    )

    first = rs.run("test-run-1")
    first_run_json = json.loads((results_root / "run.json").read_text())
    first_row_counts = json.loads((results_root / "row_counts.json").read_text())

    second = rs.run("test-run-2")
    second_run_json = json.loads((results_root / "run.json").read_text())
    second_row_counts = json.loads((results_root / "row_counts.json").read_text())

    assert first.output_count == second.output_count == 3
    assert first.issue_count == second.issue_count == 0
    # Same input -> identical hashes and counts across separate runs.
    assert (
        first_run_json["register_sources"]["input_hashes"]
        == second_run_json["register_sources"]["input_hashes"]
    )
    assert first_row_counts["register_sources"]["objects"] == second_row_counts[
        "register_sources"
    ]["objects"]


def test_run_raises_on_unsafe_archive_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, zip_bytes = _stage_fake_manifest(
        tmp_path, flaws={"google_qec": "unsafe_path"}
    )
    results_root = tmp_path / "results" / "part1"
    _patch_platform(
        monkeypatch,
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
        ("google_qec", "wrong_distance_split"),
        ("qasmbench", "missing_variant"),
        ("qasmbench", "extra_circuit"),
    ],
)
def test_run_raises_on_structural_flaw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, flaw: str
) -> None:
    manifest_path, zip_bytes = _stage_fake_manifest(tmp_path, flaws={source: flaw})
    results_root = tmp_path / "results" / "part1"
    _patch_platform(
        monkeypatch,
        manifest_path=manifest_path,
        zip_bytes_by_object_name=zip_bytes,
        results_root=results_root,
    )

    with pytest.raises(rs.BronzeError):
        rs.run("test-run-flaw")


def test_run_reports_unexpected_bronze_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extra Bronze object the manifest doesn't mandate should be counted
    as an issue and listed in run.json, without failing the run."""
    manifest_path, zip_bytes = _stage_fake_manifest(tmp_path)
    results_root = tmp_path / "results" / "part1"
    _patch_platform(
        monkeypatch,
        manifest_path=manifest_path,
        zip_bytes_by_object_name=zip_bytes,
        results_root=results_root,
    )

    stray_name = "bronze/.DS_Store"
    zip_bytes[stray_name] = b"stray"

    def _fake_bronze_inventory(settings):  # noqa: ARG001
        return [(name, 0) for name in zip_bytes]

    monkeypatch.setattr(rs, "bronze_inventory", _fake_bronze_inventory)

    result = rs.run("test-run-unexpected")
    run_json = json.loads((results_root / "run.json").read_text())

    assert result.output_count == 3
    assert result.issue_count == 1
    assert run_json["register_sources"]["unexpected_bronze_objects"] == [stray_name]