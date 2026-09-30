from quantum_lake_student.models import StageResult, stable_record_hash


def test_record_hash_is_independent_of_dictionary_order() -> None:
    assert stable_record_hash({"a": 1, "b": [2, 3]}) == stable_record_hash(
        {"b": [2, 3], "a": 1}
    )


def test_stage_result_records_completion_time() -> None:
    result = StageResult(stage="test", run_id="run-1")
    result.finish()
    assert result.finished_at is not None
    assert result.finished_at >= result.started_at



def test_revision_is_read_from_a_loose_branch_ref(tmp_path) -> None:
    from quantum_lake_student.models import revision_from_git_dir

    (tmp_path / "refs" / "heads").mkdir(parents=True)
    (tmp_path / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert revision_from_git_dir(tmp_path) == "a" * 40


def test_revision_is_read_from_packed_refs_and_detached_head(tmp_path) -> None:
    from quantum_lake_student.models import revision_from_git_dir

    (tmp_path / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n" + "b" * 40 + " refs/heads/main\n"
    )
    assert revision_from_git_dir(tmp_path) == "b" * 40

    (tmp_path / "HEAD").write_text("c" * 40 + "\n")
    assert revision_from_git_dir(tmp_path) == "c" * 40


def test_revision_is_unknown_without_a_git_dir(tmp_path) -> None:
    from quantum_lake_student.models import revision_from_git_dir

    assert revision_from_git_dir(tmp_path / "missing") == "unknown"
