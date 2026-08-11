from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from pynst import (
    GenericSweepDataset,
    OnDiskChunkManager,
    StorageCommitError,
    SweepManager,
)


def _grid(*values: Any, name: str = "point") -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [(value,) for value in values],
        names=[name],
    )


def _frame(value: float = 1.0, *, column: str = "value") -> pd.DataFrame:
    return pd.DataFrame(
        {column: [value]},
        index=pd.Index([0], name="sample"),
    )


def _mapping_result(value: float = 1.0) -> dict[str, pd.DataFrame]:
    return {"measurement": _frame(value)}


def _clear_log_entries(log_file: Path) -> None:
    """Keep the comments and header while removing all result records."""
    lines = log_file.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    header_seen = False
    for line in lines:
        if line.startswith("#"):
            kept.append(line)
        elif not header_seen:
            kept.append(line)
            header_seen = True
    log_file.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _snapshot_tree(root: Path) -> dict[str, bytes | None]:
    """Return a byte-exact snapshot, including empty directories."""
    snapshot: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = path.read_bytes() if path.is_file() else None
    return snapshot


def _assert_resume_rejected_before_callback(
    *,
    tmp_path: Path,
    meas_name: str,
    ivars: pd.MultiIndex,
) -> None:
    calls: list[dict[str, Any]] = []
    critical_errors: list[Exception] = []
    raised: BaseException | None = None

    def measurement(params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        calls.append(dict(params))
        return _mapping_result()

    try:
        resumed = SweepManager(
            measurement_func=measurement,
            ivars=ivars,
            meas_name=meas_name,
            output_root=tmp_path,
            resume=True,
            critical_callback=critical_errors.append,
        )
        try:
            resumed.run()
        except BaseException as error:  # rejection may happen at run entry
            raised = error
    except BaseException as error:  # or eagerly during resume reconciliation
        raised = error

    assert not calls, "unsafe resume reached the measurement callback"
    assert raised is not None or critical_errors, (
        "unsafe resume was neither rejected nor reported as critical"
    )


@pytest.mark.parametrize(
    "meas_name_factory",
    [
        lambda root: "",
        lambda root: ".",
        lambda root: "..",
        lambda root: "nested/name",
        lambda root: r"nested\name",
        lambda root: str(root.parent / "absolute-output"),
    ],
    ids=["empty", "dot", "parent", "slash", "backslash", "absolute"],
)
def test_meas_name_must_be_one_safe_relative_component(
    tmp_path: Path,
    meas_name_factory: Callable[[Path], str],
) -> None:
    output_root = tmp_path / "root"
    output_root.mkdir()
    sentinel = output_root / "must-not-be-deleted.txt"
    sentinel.write_text("keep", encoding="utf-8")
    meas_name = meas_name_factory(output_root)

    with pytest.raises((TypeError, ValueError), match="name|path|component|relative"):
        SweepManager(
            measurement_func=lambda params: _mapping_result(),
            ivars=_grid(1),
            meas_name=meas_name,
            output_root=output_root,
            resume=False,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_run_directory_lock_is_exclusive_and_released_after_clean_run(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=ivars,
        meas_name="exclusive_lock",
        output_root=tmp_path,
        resume=False,
    )

    with pytest.raises((RuntimeError, OSError), match="lock|active|owner|use"):
        SweepManager(
            measurement_func=lambda params: _mapping_result(),
            ivars=ivars,
            meas_name="exclusive_lock",
            output_root=tmp_path,
            resume=True,
        )

    first.run()

    resumed = SweepManager(
        measurement_func=lambda params: pytest.fail(
            "a complete resume must not measure again"
        ),
        ivars=ivars,
        meas_name="exclusive_lock",
        output_root=tmp_path,
        resume=True,
    )
    resumed.run()


def test_valid_orphan_chunk_is_reconciled_from_its_manifest(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(7.0),
        ivars=ivars,
        meas_name="orphan_reconciliation",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    first.run()
    assert len(list(first.output_dir.glob("chunk_*.h5"))) == 1

    # Simulate power loss after atomic chunk publication but before the log
    # record was made durable.
    _clear_log_entries(first.log_file)
    calls: list[int] = []

    resumed = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or _mapping_result(99.0),
        ivars=ivars,
        meas_name="orphan_reconciliation",
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    resumed.run()

    assert calls == []
    result = resumed.get_results()
    assert isinstance(result, dict)
    assert result["measurement"]["value"].tolist() == [7.0]
    assert "Complete" in resumed.log_file.read_text(encoding="utf-8")


def test_missing_complete_chunk_rejects_resume_before_callback(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=ivars,
        meas_name="missing_complete_chunk",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    first.run()
    chunk_file = next(first.output_dir.glob("chunk_*.h5"))
    chunk_file.unlink()

    _assert_resume_rejected_before_callback(
        tmp_path=tmp_path,
        meas_name="missing_complete_chunk",
        ivars=ivars,
    )


def test_duplicate_sequence_manifest_rejects_resume_before_callback(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=ivars,
        meas_name="duplicate_manifest",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    first.run()
    chunk_file = next(first.output_dir.glob("chunk_*.h5"))
    shutil.copy2(chunk_file, first.output_dir / "chunk_999.h5")

    _assert_resume_rejected_before_callback(
        tmp_path=tmp_path,
        meas_name="duplicate_manifest",
        ivars=ivars,
    )


def test_corrupt_chunk_rejects_resume_before_callback(tmp_path: Path) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=ivars,
        meas_name="corrupt_chunk",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    first.run()
    chunk_file = next(first.output_dir.glob("chunk_*.h5"))
    chunk_file.write_bytes(b"this is not an HDF5 file")

    _assert_resume_rejected_before_callback(
        tmp_path=tmp_path,
        meas_name="corrupt_chunk",
        ivars=ivars,
    )


def test_keyboard_interrupt_flushes_accepted_data_reraises_and_unlocks(
    tmp_path: Path,
) -> None:
    ivars = _grid(1, 2, 3)
    first_calls: list[int] = []

    def interrupted_measurement(params: dict[str, Any]):
        point = params["point"]
        first_calls.append(point)
        if point == 2:
            raise KeyboardInterrupt("operator abort")
        return _mapping_result(float(point))

    first = SweepManager(
        measurement_func=interrupted_measurement,
        ivars=ivars,
        meas_name="keyboard_interrupt",
        output_root=tmp_path,
        chunk_size=10,
        resume=False,
    )

    with pytest.raises(KeyboardInterrupt, match="operator abort"):
        first.run()

    assert first_calls == [1, 2]
    chunks = list(first.output_dir.glob("chunk_*.h5"))
    assert len(chunks) == 1
    with pd.HDFStore(chunks[0], mode="r") as store:
        stored = store["measurement"]
    assert stored.index.get_level_values("point").tolist() == [1]

    resumed_calls: list[int] = []
    resumed = SweepManager(
        measurement_func=lambda params: resumed_calls.append(params["point"])
        or _mapping_result(float(params["point"])),
        ivars=ivars,
        meas_name="keyboard_interrupt",
        output_root=tmp_path,
        chunk_size=10,
        resume=True,
    )
    resumed.run()

    assert resumed_calls == [2, 3]
    results = resumed.get_results()
    assert isinstance(results, dict)
    assert set(results["measurement"].index.get_level_values("point")) == {
        1,
        2,
        3,
    }


def test_storage_failure_is_critical_and_stops_before_next_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[int] = []
    critical_errors: list[Exception] = []

    def measurement(params: dict[str, Any]):
        calls.append(params["point"])
        return _mapping_result()

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1, 2),
        meas_name="critical_storage_error",
        output_root=tmp_path,
        resume=False,
        critical_callback=critical_errors.append,
    )

    def fail_storage(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manager.data_manager, "add_worker_data", fail_storage)
    raised: BaseException | None = None
    try:
        manager.run()
    except BaseException as error:
        raised = error

    captured = capsys.readouterr()
    assert calls == [1]
    assert raised is not None or critical_errors
    assert "Measurement run completed" not in captured.out
    assert "\tFailed" not in manager.log_file.read_text(encoding="utf-8")


def test_failed_points_have_explicit_incomplete_run_semantics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def measurement(params: dict[str, Any]):
        if params["point"] == 1:
            raise RuntimeError("recoverable failure")
        return _mapping_result()

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1, 2),
        meas_name="incomplete_run",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    outcome = manager.run()
    captured = capsys.readouterr()

    if isinstance(outcome, bool):
        assert outcome is False
    combined_output = (captured.out + captured.err).lower()
    assert "measurement run completed" not in captured.out.lower()
    assert "incomplete" in combined_output or "failed" in combined_output


@pytest.mark.parametrize("merge_method", ["merge", "partial_merge"])
def test_merge_requires_complete_run_unless_explicitly_overridden(
    tmp_path: Path,
    merge_method: str,
) -> None:
    def measurement(params: dict[str, Any]):
        if params["point"] == 2:
            raise RuntimeError("point intentionally failed")
        return _mapping_result(float(params["point"]))

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1, 2),
        meas_name=f"incomplete_{merge_method}",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    manager.run()
    merge = getattr(manager, merge_method)
    target = tmp_path / f"{merge_method}.h5"

    with pytest.raises(RuntimeError, match="complete|incomplete|failed"):
        merge(target)
    assert not target.exists()

    merge(target, require_complete=False)
    assert target.is_file()
    with pd.HDFStore(target, mode="r") as store:
        stored = store["measurement"]
    assert stored.index.get_level_values("point").tolist() == [1]


def test_partial_merge_drop_columns_updates_block_metadata(
    tmp_path: Path,
) -> None:
    def measurement(params: dict[str, Any]):
        return {
            "measurement": pd.DataFrame(
                {"keep": [1.0], "drop": [2.0]},
                index=pd.Index([0], name="sample"),
            )
        }

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1),
        meas_name="projected_metadata",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    manager.run()
    target = tmp_path / "projected.h5"
    manager.partial_merge(
        target,
        drop_columns={"measurement": ["drop"]},
    )

    dataset = GenericSweepDataset(target)
    block = dataset.get_block_metadata("measurement")
    assert block.dvars == ["keep"]
    assert "drop" not in block.variables
    dataset.validate_storage(deep=True)


@pytest.mark.parametrize(
    "empty_result",
    [{}, [None], (None, None)],
    ids=["empty_mapping", "one_none_slot", "only_none_slots"],
)
def test_empty_measurement_result_is_critical(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    empty_result: Any,
) -> None:
    calls: list[int] = []
    critical_errors: list[Exception] = []

    def measurement(params: dict[str, Any]):
        calls.append(params["point"])
        return empty_result

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1, 2),
        meas_name=f"empty_result_{len(calls)}_{type(empty_result).__name__}",
        output_root=tmp_path,
        resume=False,
        critical_callback=critical_errors.append,
    )
    raised: BaseException | None = None
    try:
        manager.run()
    except BaseException as error:
        raised = error

    captured = capsys.readouterr()
    assert calls == [1]
    assert raised is not None or critical_errors
    assert "Measurement run completed" not in captured.out
    assert not list(manager.output_dir.glob("chunk_*.h5"))


def test_get_results_without_nan_preserves_list_positions(
    tmp_path: Path,
) -> None:
    def measurement(params: dict[str, Any]):
        return [
            _frame(1.0, column="left"),
            _frame(float("nan"), column="empty"),
            _frame(3.0, column="right"),
        ]

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1),
        meas_name="list_positions",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    manager.run()

    results = manager.get_results(include_nan=False)
    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None
    assert results[0]["left"].tolist() == [1.0]
    assert results[2]["right"].tolist() == [3.0]


def test_resume_preserves_initial_global_metadata_and_created_at(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=ivars,
        meas_name="metadata_resume",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        metadata={"operator": "Ada", "setup_id": "fixture-A"},
    )
    created_at = first.metadata["created_at"]
    first.run()

    resumed = SweepManager(
        measurement_func=lambda params: pytest.fail(
            "complete resume must not call hardware"
        ),
        ivars=ivars,
        meas_name="metadata_resume",
        output_root=tmp_path,
        resume=True,
    )
    assert resumed.metadata["created_at"] == created_at
    assert resumed.metadata["operator"] == "Ada"
    assert resumed.metadata["setup_id"] == "fixture-A"
    resumed.run()

    target = tmp_path / "metadata_resume.h5"
    resumed.merge(target)
    metadata = SweepManager.read_merged_metadata(target)
    assert metadata.created_at == created_at
    assert metadata.extra["operator"] == "Ada"
    assert metadata.extra["setup_id"] == "fixture-A"


def test_mixed_object_sweep_grid_is_rejected_before_callback_or_output(
    tmp_path: Path,
) -> None:
    ivars = _grid(1, "1", "tab\tvalue")
    calls: list[Any] = []

    with pytest.raises(TypeError, match="mixed|object|homogeneous"):
        SweepManager(
            measurement_func=lambda params: calls.append(params["point"])
            or _mapping_result(),
            ivars=ivars,
            meas_name="mixed_object_grid",
            output_root=tmp_path,
            resume=False,
        )

    assert calls == []
    assert not (tmp_path / "mixed_object_grid").exists()


def test_sequence_identity_handles_homogeneous_strings_and_tabs(
    tmp_path: Path,
) -> None:
    ivars = _grid("plain", "retry", "tab\tvalue")

    def first_measurement(params: dict[str, Any]):
        if params["point"] == "retry":
            raise RuntimeError("retry this sequence later")
        return _mapping_result()

    first = SweepManager(
        measurement_func=first_measurement,
        ivars=ivars,
        meas_name="string_sequence_identity",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    first.run()

    resumed_calls: list[Any] = []

    def resumed_measurement(params: dict[str, Any]):
        resumed_calls.append(params["point"])
        return _mapping_result()

    resumed = SweepManager(
        measurement_func=resumed_measurement,
        ivars=ivars,
        meas_name="string_sequence_identity",
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    resumed.run()

    assert resumed_calls == ["retry"]
    result = resumed.get_results()
    assert isinstance(result, dict)
    stored_values = result["measurement"].index.get_level_values("point")
    assert set(stored_values) == {"plain", "retry", "tab\tvalue"}

    manifest = json.loads(resumed.manifest_file.read_text(encoding="utf-8"))
    committed_sequences = sorted(
        sequence_index
        for chunk in manifest["chunks"]
        for sequence_index in chunk["sequence_indices"]
    )
    assert committed_sequences == [0, 1, 2]

    target = tmp_path / "string_sequence_identity.h5"
    resumed.merge(target)
    merged = GenericSweepDataset(target)
    assert set(
        merged.get_block("measurement").index.get_level_values("point")
    ) == {"plain", "retry", "tab\tvalue"}
    merged.validate_storage(deep=True)


def test_remove_chunks_archives_run_and_resume_is_rejected(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=ivars,
        meas_name="archived_run",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    first.run()
    first.merge(tmp_path / "archived.h5", remove_chunks=True)
    assert not list(first.output_dir.glob("chunk_*.h5"))

    _assert_resume_rejected_before_callback(
        tmp_path=tmp_path,
        meas_name="archived_run",
        ivars=ivars,
    )


def test_deep_dataset_validation_checks_rows_beyond_the_first(
    tmp_path: Path,
) -> None:
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(
            float(params["point"])
        ),
        ivars=_grid(1, 2),
        meas_name="deep_validation",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    manager.run()
    target = tmp_path / "deep_validation.h5"
    manager.merge(target)

    with pd.HDFStore(target, mode="a") as store:
        frame = store["measurement"]
        corrupt = pd.concat([frame, frame.tail(1)])
        store.put("measurement", corrupt, format="fixed")

    dataset = GenericSweepDataset(target, validate=False)
    with pytest.raises(ValueError, match="duplicate"):
        dataset.validate_storage(deep=True)


def test_package_declares_python_39_for_removeprefix_usage() -> None:
    setup_cfg = Path(__file__).resolve().parents[1] / "setup.cfg"
    text = setup_cfg.read_text(encoding="utf-8")
    assert "python_requires = >=3.10" in text


def test_run_after_close_reacquires_lock_and_respects_active_owner(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or _mapping_result(),
        ivars=ivars,
        meas_name="close_then_run",
        output_root=tmp_path,
        resume=False,
    )
    first.close()

    blocker = SweepManager(
        measurement_func=lambda params: pytest.fail(
            "the blocker must not execute a measurement"
        ),
        ivars=ivars,
        meas_name="close_then_run",
        output_root=tmp_path,
        resume=True,
    )
    with pytest.raises((RuntimeError, OSError), match="lock|active|owner|use"):
        first.run()
    assert calls == []

    blocker.close()
    assert first.run() is True
    assert calls == [1]


def test_stale_manager_rejects_run_uuid_replacement_before_callback(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    stale_calls: list[int] = []
    stale = SweepManager(
        measurement_func=lambda params: stale_calls.append(params["point"])
        or _mapping_result(99.0),
        ivars=ivars,
        meas_name="replaced_run_uuid",
        output_root=tmp_path,
        resume=False,
    )
    stale_uuid = stale.run_uuid
    stale.close()

    replacement_calls: list[int] = []
    replacement = SweepManager(
        measurement_func=lambda params: replacement_calls.append(
            params["point"]
        )
        or _mapping_result(5.0),
        ivars=ivars,
        meas_name="replaced_run_uuid",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
    )
    assert replacement.run_uuid != stale_uuid
    replacement.close()
    assert replacement.run() is True
    assert replacement_calls == [1]
    replacement_uuid = replacement.run_uuid
    before = _snapshot_tree(replacement.output_dir)

    with pytest.raises(RuntimeError, match="UUID|replaced|different run"):
        stale.run()

    assert stale_calls == []
    assert _snapshot_tree(replacement.output_dir) == before
    contract = json.loads(
        replacement.contract_file.read_text(encoding="utf-8")
    )
    manifest = json.loads(
        replacement.manifest_file.read_text(encoding="utf-8")
    )
    assert contract["run_uuid"] == replacement_uuid
    assert manifest["run_uuid"] == replacement_uuid


def test_update_metadata_after_run_reacquires_lock_and_persists(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=ivars,
        meas_name="metadata_locking",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert first.run() is True

    blocker = SweepManager(
        measurement_func=lambda params: pytest.fail(
            "the complete run must not measure again"
        ),
        ivars=ivars,
        meas_name="metadata_locking",
        output_root=tmp_path,
        resume=True,
    )
    with pytest.raises((RuntimeError, OSError), match="lock|active|owner|use"):
        first.update_metadata(operator="must-not-be-written")
    blocker.close()

    first.update_metadata(
        operator="Grace",
        blocks={
            "measurement": {
                "description": "persisted after the measurement run"
            }
        },
    )
    resumed = SweepManager(
        measurement_func=lambda params: pytest.fail(
            "the complete run must not measure again"
        ),
        ivars=ivars,
        meas_name="metadata_locking",
        output_root=tmp_path,
        resume=True,
    )
    assert resumed.metadata["operator"] == "Grace"
    assert (
        resumed._user_block_metadata["measurement"]["description"]
        == "persisted after the measurement run"
    )
    resumed.close()


@pytest.mark.parametrize("merge_method", ["merge", "partial_merge"])
@pytest.mark.parametrize(
    "target_name",
    [
        "contract",
        "manifest",
        "log",
        "traceback",
        "output_dir",
        "chunk",
    ],
)
def test_merge_rejects_run_artifact_targets_before_mutation(
    tmp_path: Path,
    merge_method: str,
    target_name: str,
) -> None:
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=_grid(1),
        meas_name=f"reserved_{merge_method}_{target_name}",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    manager.errlog_file.write_text("diagnostic sentinel\n", encoding="utf-8")
    chunk_file = next(manager.output_dir.glob("chunk_*.h5"))
    targets = {
        "contract": manager.contract_file,
        "manifest": manager.manifest_file,
        "log": manager.log_file,
        "traceback": manager.errlog_file,
        "output_dir": manager.output_dir,
        "chunk": chunk_file,
    }
    before = _snapshot_tree(manager.output_dir)
    merge = getattr(manager, merge_method)
    overwrite_kwarg = (
        {"overwrite": True}
        if merge_method == "merge"
        else {"force_merge_into_existing": True}
    )

    with pytest.raises((TypeError, ValueError), match="run|artifact|chunk|file"):
        merge(targets[target_name], **overwrite_kwarg)

    assert _snapshot_tree(manager.output_dir) == before


def test_empty_multiindex_is_rejected_before_creating_a_run(
    tmp_path: Path,
) -> None:
    empty = pd.MultiIndex(
        levels=[pd.Index([], dtype="int64")],
        codes=[[]],
        names=["point"],
    )
    with pytest.raises(ValueError, match="at least one|empty|combination"):
        SweepManager(
            measurement_func=lambda params: _mapping_result(),
            ivars=empty,
            meas_name="empty_grid",
            output_root=tmp_path,
            resume=False,
        )
    assert not (tmp_path / "empty_grid").exists()


def test_v3_chunks_refuse_in_place_incomplete_row_removal(
    tmp_path: Path,
) -> None:
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=_grid(1),
        meas_name="immutable_committed_chunk",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    chunk_file = next(manager.output_dir.glob("chunk_*.h5"))
    before = chunk_file.read_bytes()

    with pytest.raises(RuntimeError, match="in-place|committed|immutable"):
        manager.data_manager.remove_incomplete_params_from_chunks(
            [("1",)],
            ["point"],
        )

    assert chunk_file.read_bytes() == before


def _make_listed_chunk_an_orphan(manager: SweepManager) -> Path:
    manifest = json.loads(manager.manifest_file.read_text(encoding="utf-8"))
    assert len(manifest["chunks"]) == 1
    manifest["chunks"] = []
    manager.manifest_file.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return next(manager.output_dir.glob("chunk_*.h5"))


@pytest.mark.parametrize("corruption", ["missing_run_uuid", "wrong_dtype"])
def test_v3_orphan_requires_commit_metadata_and_schema(
    tmp_path: Path,
    corruption: str,
) -> None:
    ivars = _grid(1)
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(1.25),
        ivars=ivars,
        meas_name=f"invalid_v3_orphan_{corruption}",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    chunk_file = _make_listed_chunk_an_orphan(manager)

    with pd.HDFStore(chunk_file, mode="a") as store:
        if corruption == "missing_run_uuid":
            delattr(store.root._v_attrs, "pynst_run_uuid")
        else:
            frame = store["measurement"]
            frame["value"] = frame["value"].astype(str)
            store.put(
                "measurement",
                frame,
                format="table",
                data_columns=list(frame.index.names),
            )

    _assert_resume_rejected_before_callback(
        tmp_path=tmp_path,
        meas_name=f"invalid_v3_orphan_{corruption}",
        ivars=ivars,
    )


def test_v2_like_run_upgrades_once_and_survives_a_second_resume(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    seed = SweepManager(
        measurement_func=lambda params: _mapping_result(4.0),
        ivars=ivars,
        meas_name="legacy_v2_upgrade",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert seed.run() is True

    contract = json.loads(seed.contract_file.read_text(encoding="utf-8"))
    contract["format_version"] = 2
    contract.pop("run_uuid", None)
    seed.contract_file.write_text(
        json.dumps(contract, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seed.manifest_file.unlink()
    seed.log_file.write_text(
        '# ["point"]\n'
        "point\tchunk_file\tstatus\n"
        "1\tchunk_0.h5\tComplete\n",
        encoding="utf-8",
    )
    chunk_file = next(seed.output_dir.glob("chunk_*.h5"))
    with pd.HDFStore(chunk_file, mode="a") as store:
        for name in (
            "pynst_chunk_index",
            "pynst_sequence_indices_json",
            "pynst_block_names_json",
            "pynst_run_uuid",
            "pynst_contract_sha256",
        ):
            try:
                delattr(store.root._v_attrs, name)
            except AttributeError:
                pass

    first_resume_calls: list[int] = []
    upgraded = SweepManager(
        measurement_func=lambda params: first_resume_calls.append(
            params["point"]
        )
        or _mapping_result(99.0),
        ivars=ivars,
        meas_name="legacy_v2_upgrade",
        output_root=tmp_path,
        resume=True,
    )
    assert upgraded.run() is True
    assert first_resume_calls == []
    upgraded_contract = json.loads(
        upgraded.contract_file.read_text(encoding="utf-8")
    )
    assert upgraded_contract["format_version"] == 3
    assert upgraded.manifest_file.is_file()

    second_resume_calls: list[int] = []
    resumed_again = SweepManager(
        measurement_func=lambda params: second_resume_calls.append(
            params["point"]
        )
        or _mapping_result(100.0),
        ivars=ivars,
        meas_name="legacy_v2_upgrade",
        output_root=tmp_path,
        resume=True,
    )
    assert resumed_again.run() is True
    assert second_resume_calls == []
    result = resumed_again.get_results()
    assert isinstance(result, dict)
    assert result["measurement"]["value"].tolist() == [4.0]


def test_manifest_commit_failure_after_final_rename_is_terminal_but_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ivars = _grid(1)
    critical_errors: list[Exception] = []
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(8.0),
        ivars=ivars,
        meas_name="post_rename_manifest_failure",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        critical_callback=critical_errors.append,
    )
    original_write_manifest = manager._write_run_manifest

    def fail_manifest_commit_after_final_rename() -> None:
        final_chunk_exists = bool(
            list(manager.output_dir.glob("chunk_*.h5"))
        )
        chunk_is_pending_in_memory = bool(manager._manifest.get("chunks"))
        if final_chunk_exists and chunk_is_pending_in_memory:
            raise OSError("simulated manifest fsync failure")
        original_write_manifest()

    monkeypatch.setattr(
        manager,
        "_write_run_manifest",
        fail_manifest_commit_after_final_rename,
    )

    assert manager.run() is False
    assert len(critical_errors) == 1
    assert isinstance(critical_errors[0], StorageCommitError)
    chunks = list(manager.output_dir.glob("chunk_*.h5"))
    assert len(chunks) == 1
    assert manager.data_manager.worker_count == 1
    assert manager.data_manager.pending_sequence_indices == [0]
    assert manager.data_manager.df_blocks

    with pytest.raises(StorageCommitError, match="terminal|ambiguous"):
        manager.get_results()
    with pytest.raises(StorageCommitError, match="terminal|ambiguous"):
        manager.update_metadata(operator="must-not-be-written")
    merge_target = tmp_path / "must-not-be-created.h5"
    with pytest.raises(StorageCommitError, match="terminal|ambiguous"):
        manager.merge(merge_target)
    assert not merge_target.exists()

    resumed_calls: list[int] = []
    resumed = SweepManager(
        measurement_func=lambda params: resumed_calls.append(params["point"])
        or _mapping_result(99.0),
        ivars=ivars,
        meas_name="post_rename_manifest_failure",
        output_root=tmp_path,
        resume=True,
    )
    assert resumed.run() is True
    assert resumed_calls == []
    result = resumed.get_results()
    assert isinstance(result, dict)
    assert result["measurement"]["value"].tolist() == [8.0]


def test_v3_resume_repairs_non_utf8_diagnostic_log_from_manifest(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(6.0),
        ivars=ivars,
        meas_name="corrupt_diagnostic_log",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert first.run() is True
    first.log_file.write_bytes(b"\xff\xfe\x80not-valid-utf8\x00")

    calls: list[int] = []
    resumed = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or _mapping_result(99.0),
        ivars=ivars,
        meas_name="corrupt_diagnostic_log",
        output_root=tmp_path,
        resume=True,
    )
    assert resumed.run() is True
    assert calls == []

    repaired = resumed.log_file.read_text(encoding="utf-8")
    assert "Complete" in repaired
    assert "sequence_index" in repaired
    assert "chunk_0.h5" in repaired


def test_optional_mapping_none_is_preserved_in_previous_and_results(
    tmp_path: Path,
) -> None:
    previous_results: list[Any] = []

    def measurement(params: dict[str, Any], previous_result: Any):
        previous_results.append(previous_result)
        return {
            "required": _frame(float(params["point"])),
            "optional": None,
        }

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1, 2),
        meas_name="optional_mapping_slot",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        provide_previous_result=True,
    )
    assert manager.run() is True

    assert previous_results[0] is None
    assert isinstance(previous_results[1], dict)
    assert set(previous_results[1]) == {"required", "optional"}
    assert previous_results[1]["optional"] is None
    assert isinstance(previous_results[1]["required"], pd.DataFrame)
    assert (
        previous_results[1]["required"]
        .index.get_level_values("point")
        .tolist()
        == [1]
    )

    results = manager.get_results()
    assert isinstance(results, dict)
    assert set(results) == {"required", "optional"}
    assert results["optional"] is None
    assert isinstance(results["required"], pd.DataFrame)
    assert results["required"].index.get_level_values("point").tolist() == [
        1,
        2,
    ]


def test_resumed_semantic_metadata_survives_another_resume_and_merge(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(3.0),
        ivars=ivars,
        meas_name="resumed_semantic_metadata",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert first.run() is True

    editor = SweepManager(
        measurement_func=lambda params: pytest.fail(
            "complete run must not invoke the measurement callback"
        ),
        ivars=ivars,
        meas_name="resumed_semantic_metadata",
        output_root=tmp_path,
        resume=True,
    )
    editor.update_metadata(
        blocks={
            "measurement": {
                "description": "reviewed measurement block",
                "variables": {"value": {"unit": "V"}},
            }
        }
    )
    editor.close()

    merger = SweepManager(
        measurement_func=lambda params: pytest.fail(
            "complete run must not invoke the measurement callback"
        ),
        ivars=ivars,
        meas_name="resumed_semantic_metadata",
        output_root=tmp_path,
        resume=True,
    )
    target = tmp_path / "resumed_semantic_metadata.h5"
    merger.merge(target)

    dataset = GenericSweepDataset(target)
    block = dataset.get_block_metadata("measurement")
    assert block.description == "reviewed measurement block"
    assert block.variables["value"].unit == "V"
    dataset.validate_storage(deep=True)


def test_fresh_contract_hash_matches_exact_file_and_manifest_bytes(
    tmp_path: Path,
) -> None:
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(2.0),
        ivars=_grid(1),
        meas_name="exact_windows_contract_hash",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )

    initial_hash = hashlib.sha256(manager.contract_file.read_bytes()).hexdigest()
    initial_manifest = json.loads(
        manager.manifest_file.read_text(encoding="utf-8")
    )
    assert manager.contract_sha256 == initial_hash
    assert initial_manifest["contract_sha256"] == initial_hash

    assert manager.run() is True
    final_hash = hashlib.sha256(manager.contract_file.read_bytes()).hexdigest()
    final_manifest = json.loads(
        manager.manifest_file.read_text(encoding="utf-8")
    )
    assert manager.contract_sha256 == final_hash
    assert final_manifest["contract_sha256"] == final_hash
    with pd.HDFStore(
        next(manager.output_dir.glob("chunk_*.h5")), mode="r"
    ) as store:
        assert str(store.root._v_attrs.pynst_contract_sha256) == final_hash


def test_first_schema_contract_replace_survives_manifest_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ivars = _grid(1)
    calls: list[int] = []
    manager = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or _mapping_result(7.0),
        ivars=ivars,
        meas_name="first_schema_manifest_failure",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    old_manifest_hash = json.loads(
        manager.manifest_file.read_text(encoding="utf-8")
    )["contract_sha256"]
    original_write_manifest = manager._write_run_manifest
    injected = False

    def fail_first_schema_manifest_commit() -> None:
        nonlocal injected
        contract = json.loads(
            manager.contract_file.read_text(encoding="utf-8")
        )
        if not injected and contract.get("result_schema") is not None:
            injected = True
            raise OSError("simulated first-schema manifest commit failure")
        original_write_manifest()

    with monkeypatch.context() as patcher:
        patcher.setattr(
            manager,
            "_write_run_manifest",
            fail_first_schema_manifest_commit,
        )
        assert manager.run() is False

    assert calls == [1]
    assert injected
    assert not list(manager.output_dir.glob("chunk_*.h5"))
    replaced_contract = json.loads(
        manager.contract_file.read_text(encoding="utf-8")
    )
    assert replaced_contract["result_schema"] is not None
    replaced_hash = hashlib.sha256(
        manager.contract_file.read_bytes()
    ).hexdigest()
    stale_manifest = json.loads(
        manager.manifest_file.read_text(encoding="utf-8")
    )
    assert stale_manifest["contract_sha256"] == old_manifest_hash
    assert stale_manifest["contract_sha256"] != replaced_hash

    resumed_calls: list[int] = []
    resumed = SweepManager(
        measurement_func=lambda params: resumed_calls.append(params["point"])
        or _mapping_result(7.0),
        ivars=ivars,
        meas_name="first_schema_manifest_failure",
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    assert resumed.run() is True
    assert resumed_calls == [1]
    repaired_manifest = json.loads(
        resumed.manifest_file.read_text(encoding="utf-8")
    )
    assert repaired_manifest["contract_sha256"] == replaced_hash
    assert len(repaired_manifest["chunks"]) == 1


def test_global_sweep_level_dtypes_survive_merge_and_archive(
    tmp_path: Path,
) -> None:
    ivars = pd.MultiIndex.from_arrays(
        [
            pd.Series([1, 2], dtype="int32"),
            pd.Series([0.5, 1.5], dtype="float32"),
            pd.Categorical(
                ["cold", "hot"],
                categories=["cold", "nominal", "hot"],
                ordered=True,
            ),
        ],
        names=["integer_level", "float_level", "category_level"],
    )
    expected_dtypes = {
        "integer_level": "int32",
        "float_level": "float32",
        "category_level": "category",
    }
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(1.0),
        ivars=ivars,
        meas_name="global_level_dtypes",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    target = tmp_path / "global_level_dtypes.h5"
    manager.merge(target, remove_chunks=True)

    assert not list(manager.output_dir.glob("chunk_*.h5"))
    dataset = GenericSweepDataset(target)
    block = dataset.get_block_metadata("measurement")
    stored = dataset.get_block("measurement")
    for name, expected_dtype in expected_dtypes.items():
        assert block.variables[name].dtype == expected_dtype
        assert str(stored.index.get_level_values(name).dtype) == expected_dtype
    dataset.validate_storage(deep=True)


def test_v2_upgrade_crash_after_contract_replace_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ivars = _grid(1)
    seed = SweepManager(
        measurement_func=lambda params: _mapping_result(4.5),
        ivars=ivars,
        meas_name="legacy_upgrade_contract_replace_crash",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert seed.run() is True

    contract = json.loads(seed.contract_file.read_text(encoding="utf-8"))
    contract["format_version"] = 2
    contract.pop("run_uuid", None)
    seed.contract_file.write_text(
        json.dumps(contract, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seed.manifest_file.unlink()
    seed.log_file.write_text(
        '# ["point"]\n'
        "point\tchunk_file\tstatus\n"
        "1\tchunk_0.h5\tComplete\n",
        encoding="utf-8",
    )
    chunk_file = next(seed.output_dir.glob("chunk_*.h5"))
    with pd.HDFStore(chunk_file, mode="a") as store:
        for name in (
            "pynst_chunk_index",
            "pynst_sequence_indices_json",
            "pynst_block_names_json",
            "pynst_run_uuid",
            "pynst_contract_sha256",
        ):
            try:
                delattr(store.root._v_attrs, name)
            except AttributeError:
                pass

    original_write_manifest = SweepManager._write_run_manifest
    injected = False

    def fail_first_migrated_manifest_commit(manager: SweepManager) -> None:
        nonlocal injected
        migrated_contract = json.loads(
            manager.contract_file.read_text(encoding="utf-8")
        )
        persisted_manifest = (
            json.loads(manager.manifest_file.read_text(encoding="utf-8"))
            if manager.manifest_file.exists()
            else {}
        )
        if (
            manager.meas_name == "legacy_upgrade_contract_replace_crash"
            and not injected
            and migrated_contract["format_version"] == 3
            and manager._manifest.get("state") == "active"
            and persisted_manifest.get("state") == "migrating_legacy"
        ):
            injected = True
            raise OSError("simulated crash after migrated contract replace")
        original_write_manifest(manager)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            SweepManager,
            "_write_run_manifest",
            fail_first_migrated_manifest_commit,
        )
        with pytest.raises(OSError, match="migrated contract replace"):
            SweepManager(
                measurement_func=lambda params: pytest.fail(
                    "migration must not invoke the measurement callback"
                ),
                ivars=ivars,
                meas_name="legacy_upgrade_contract_replace_crash",
                output_root=tmp_path,
                resume=True,
            )

    assert injected
    migrated_contract = json.loads(seed.contract_file.read_text(encoding="utf-8"))
    assert migrated_contract["format_version"] == 3
    interrupted_manifest = json.loads(
        seed.manifest_file.read_text(encoding="utf-8")
    )
    assert interrupted_manifest["state"] == "migrating_legacy"
    assert "migration" in interrupted_manifest

    resumed_calls: list[int] = []
    resumed = SweepManager(
        measurement_func=lambda params: resumed_calls.append(params["point"])
        or _mapping_result(99.0),
        ivars=ivars,
        meas_name="legacy_upgrade_contract_replace_crash",
        output_root=tmp_path,
        resume=True,
    )
    assert resumed.run() is True
    assert resumed_calls == []
    result = resumed.get_results()
    assert isinstance(result, dict)
    assert result["measurement"]["value"].tolist() == [4.5]


@pytest.mark.parametrize(
    ("changed_categories", "changed_ordered"),
    [
        (["cold", "hot", "fault"], False),
        (["cold", "nominal", "hot"], True),
    ],
    ids=["category_vocabulary", "category_ordering"],
)
def test_resume_rejects_changed_categorical_grid_semantics_before_callback(
    tmp_path: Path,
    changed_categories: list[str],
    changed_ordered: bool,
) -> None:
    name = f"categorical_grid_{changed_ordered}_{changed_categories[-1]}"
    initial = pd.MultiIndex.from_arrays(
        [
            pd.Categorical(
                ["cold", "hot"],
                categories=["cold", "nominal", "hot"],
                ordered=False,
            )
        ],
        names=["state"],
    )
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=initial,
        meas_name=name,
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True

    changed = pd.MultiIndex.from_arrays(
        [
            pd.Categorical(
                ["cold", "hot"],
                categories=changed_categories,
                ordered=changed_ordered,
            )
        ],
        names=["state"],
    )
    _assert_resume_rejected_before_callback(
        tmp_path=tmp_path,
        meas_name=name,
        ivars=changed,
    )


def _local_schema_frame(case: str, *, changed: bool) -> pd.DataFrame:
    if case == "index_dtype":
        index = pd.Index(
            pd.Series([0], dtype="int64" if changed else "int32"),
            name="sample",
        )
        return pd.DataFrame({"value": [1.0]}, index=index)
    if case == "categorical_ivar":
        index = pd.CategoricalIndex(
            pd.Categorical(
                ["near"],
                categories=(
                    ["near", "far", "invalid"]
                    if changed
                    else ["near", "far"]
                ),
                ordered=changed,
            ),
            name="position",
        )
        return pd.DataFrame({"value": [1.0]}, index=index)
    if case == "categorical_dvar":
        state = pd.Categorical(
            ["valid"],
            categories=(
                ["valid", "invalid", "unknown"]
                if changed
                else ["valid", "invalid"]
            ),
            ordered=changed,
        )
        return pd.DataFrame(
            {"state": state},
            index=pd.Index([0], name="sample"),
        )
    raise AssertionError(f"unknown test case {case!r}")


@pytest.mark.parametrize(
    "case",
    ["index_dtype", "categorical_ivar", "categorical_dvar"],
)
def test_resume_rejects_changed_local_dtype_or_category_semantics(
    tmp_path: Path,
    case: str,
) -> None:
    name = f"changed_local_schema_{case}"
    first_calls: list[int] = []

    def first_measurement(params: dict[str, Any]):
        first_calls.append(params["point"])
        if params["point"] == 2:
            raise RuntimeError("leave the second point pending")
        return {"measurement": _local_schema_frame(case, changed=False)}

    first = SweepManager(
        measurement_func=first_measurement,
        ivars=_grid(1, 2),
        meas_name=name,
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert first.run() is False
    assert first_calls == [1, 2]
    initial_chunks = list(first.output_dir.glob("chunk_*.h5"))
    assert len(initial_chunks) == 1
    initial_chunk_bytes = initial_chunks[0].read_bytes()

    resumed_calls: list[int] = []
    critical_errors: list[Exception] = []
    resumed = SweepManager(
        measurement_func=lambda params: resumed_calls.append(params["point"])
        or {"measurement": _local_schema_frame(case, changed=True)},
        ivars=_grid(1, 2),
        meas_name=name,
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
        critical_callback=critical_errors.append,
    )
    assert resumed.run() is False
    assert resumed_calls == [2]
    assert len(critical_errors) == 1
    assert "dtype" in str(critical_errors[0]).lower() or "categor" in str(
        critical_errors[0]
    ).lower()
    assert list(resumed.output_dir.glob("chunk_*.h5")) == initial_chunks
    assert initial_chunks[0].read_bytes() == initial_chunk_bytes


def test_first_tagging_name_conflict_does_not_bind_result_schema(
    tmp_path: Path,
) -> None:
    ivars = _grid(1, 2)
    critical_errors: list[Exception] = []
    first = SweepManager(
        measurement_func=lambda params: {
            "measurement": pd.DataFrame(
                {"point": [float(params["point"])]},
                index=pd.Index([0], name="sample"),
            )
        },
        ivars=ivars,
        meas_name="first_tagging_name_conflict",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        critical_callback=critical_errors.append,
    )
    assert first.run() is False
    assert len(critical_errors) == 1
    assert not list(first.output_dir.glob("chunk_*.h5"))
    contract = json.loads(first.contract_file.read_text(encoding="utf-8"))
    assert contract["result_schema"] is None
    assert contract["block_names"] is None
    assert contract["block_mode"] is None

    calls: list[int] = []
    resumed = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or _mapping_result(float(params["point"])),
        ivars=ivars,
        meas_name="first_tagging_name_conflict",
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    assert resumed.run() is True
    assert calls == [1, 2]
    result = resumed.get_results()
    assert isinstance(result, dict)
    assert result["measurement"]["value"].tolist() == [1.0, 2.0]


def test_closed_stale_manager_get_results_rejects_replaced_run_uuid(
    tmp_path: Path,
) -> None:
    ivars = _grid(1)
    stale = SweepManager(
        measurement_func=lambda params: _mapping_result(1.0),
        ivars=ivars,
        meas_name="stale_get_results_uuid",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert stale.run() is True
    stale_uuid = stale.run_uuid
    stale.close()

    replacement = SweepManager(
        measurement_func=lambda params: _mapping_result(9.0),
        ivars=ivars,
        meas_name="stale_get_results_uuid",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert replacement.run() is True
    assert replacement.run_uuid != stale_uuid
    replacement.close()
    before = _snapshot_tree(replacement.output_dir)

    with pytest.raises(RuntimeError, match="UUID|replaced|different|identity"):
        stale.get_results()

    assert _snapshot_tree(replacement.output_dir) == before


def test_partial_merge_rejects_dropping_all_dvars_without_target_mutation(
    tmp_path: Path,
) -> None:
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=_grid(1),
        meas_name="drop_every_dvar",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    target = tmp_path / "drop_every_dvar.h5"
    before = _snapshot_tree(manager.output_dir)

    with pytest.raises(ValueError, match="dependent|every|all|dvar"):
        manager.partial_merge(
            target,
            drop_columns={"measurement": ["value"]},
        )

    assert not target.exists()
    assert _snapshot_tree(manager.output_dir) == before


@pytest.mark.parametrize("merge_method", ["merge", "partial_merge"])
def test_remove_chunks_rejects_target_inside_run_directory(
    tmp_path: Path,
    merge_method: str,
) -> None:
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=_grid(1),
        meas_name=f"retirement_inside_run_{merge_method}",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    target = manager.output_dir / "archive.h5"
    before = _snapshot_tree(manager.output_dir)

    with pytest.raises(ValueError, match="run|inside|archive|retir|target"):
        getattr(manager, merge_method)(target, remove_chunks=True)

    assert not target.exists()
    assert _snapshot_tree(manager.output_dir) == before


@pytest.mark.parametrize("merge_method", ["merge", "partial_merge"])
def test_merge_preserves_preexisting_legacy_named_temporary_file(
    tmp_path: Path,
    merge_method: str,
) -> None:
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(2.5),
        ivars=_grid(1),
        meas_name=f"preserve_temp_{merge_method}",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    target = tmp_path / f"preserve_temp_{merge_method}.h5"
    legacy_temporary = Path(str(target) + ".pynst_tmp")
    sentinel = b"legitimate unrelated file\x00do not overwrite"
    legacy_temporary.write_bytes(sentinel)

    getattr(manager, merge_method)(target)

    assert target.is_file()
    assert legacy_temporary.read_bytes() == sentinel
    GenericSweepDataset(target).validate_storage(deep=True)


@pytest.mark.parametrize("block_mode", ["mapping", "list"])
def test_merge_retains_declared_none_slots_in_metadata_and_root_attrs(
    tmp_path: Path,
    block_mode: str,
) -> None:
    expected_names = (
        ["measurement", "optional"]
        if block_mode == "mapping"
        else ["block_0", "block_1"]
    )

    def measurement(params: dict[str, Any]):
        if block_mode == "mapping":
            return {"measurement": _frame(1.0), "optional": None}
        return [_frame(1.0), None]

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1),
        meas_name=f"merged_none_slots_{block_mode}",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    target = tmp_path / f"merged_none_slots_{block_mode}.h5"
    manager.merge(target)

    dataset = GenericSweepDataset(target)
    assert dataset.metadata.extra["pynst_result_block_names"] == expected_names
    assert dataset.metadata.extra["pynst_block_mode"] == block_mode
    with pd.HDFStore(target, mode="r") as store:
        root = store.root._v_attrs
        assert json.loads(str(root.pynst_block_names_json)) == expected_names
        assert str(root.pynst_block_mode) == block_mode
    dataset.validate_storage(deep=True)


def test_direct_chunk_manager_resume_restores_none_list_slots(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "direct_chunks"
    first = OnDiskChunkManager(
        chunk_size=1,
        output_dir=output_dir,
        resume=False,
    )
    first.configure_run_identity(
        run_uuid="direct-run",
        contract_sha256="direct-contract",
        block_names=["block_0", "block_1"],
    )
    first.add_worker_data(
        {"block_0": _frame(4.0)},
        metadata={
            "block_mode": "list",
            "block_names": ["block_0", "block_1"],
        },
        sequence_indices=[0],
    )
    first.finalize()

    resumed = OnDiskChunkManager(
        chunk_size=1,
        output_dir=output_dir,
        resume=True,
    )
    result = resumed.get_results()
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert result[0]["value"].tolist() == [4.0]
    assert result[1] is None


def test_index_only_dataframe_is_rejected_before_first_chunk_commit(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    critical_errors: list[Exception] = []

    def measurement(params: dict[str, Any]):
        calls.append(params["point"])
        return {
            "measurement": pd.DataFrame(
                index=pd.Index([0], name="sample")
            )
        }

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1, 2),
        meas_name="index_only_result",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        critical_callback=critical_errors.append,
    )
    assert manager.run() is False
    assert calls == [1]
    assert len(critical_errors) == 1
    assert any(
        marker in str(critical_errors[0]).lower()
        for marker in ("dependent", "empty")
    )
    assert not list(manager.output_dir.glob("chunk_*.h5"))
    contract = json.loads(manager.contract_file.read_text(encoding="utf-8"))
    assert contract["result_schema"] is None


@pytest.mark.parametrize("extension_location", ["dvar", "local_ivar"])
def test_nullable_extension_dtype_is_rejected_before_schema_binding(
    tmp_path: Path,
    extension_location: str,
) -> None:
    calls: list[int] = []
    critical_errors: list[Exception] = []

    def unsupported_frame() -> pd.DataFrame:
        if extension_location == "local_ivar":
            index = pd.Index(
                pd.array([0], dtype="Int64"),
                name="sample",
            )
            return pd.DataFrame({"value": [1.0]}, index=index)
        return pd.DataFrame(
            {"value": pd.Series([1], dtype="Int64")},
            index=pd.Index([0], name="sample"),
        )

    manager = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or {"measurement": unsupported_frame()},
        ivars=_grid(1),
        meas_name=f"unsupported_extension_{extension_location}",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        critical_callback=critical_errors.append,
    )
    assert manager.run() is False
    assert calls == [1]
    assert len(critical_errors) == 1
    assert "extension" in str(critical_errors[0]).lower()
    assert not list(manager.output_dir.glob("chunk_*"))
    contract = json.loads(manager.contract_file.read_text(encoding="utf-8"))
    assert contract["result_schema"] is None
    assert contract["block_names"] is None
    assert contract["block_mode"] is None

    corrected_calls: list[int] = []
    corrected = SweepManager(
        measurement_func=lambda params: corrected_calls.append(params["point"])
        or _mapping_result(2.0),
        ivars=_grid(1),
        meas_name=f"unsupported_extension_{extension_location}",
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    assert corrected.run() is True
    assert corrected_calls == [1]
    result = corrected.get_results()
    assert isinstance(result, dict)
    assert result["measurement"]["value"].tolist() == [2.0]


@pytest.mark.parametrize(
    ("object_location", "expected_kind"),
    [
        ("dvar", "mixed"),
        ("local_ivar", "mixed"),
        ("bytes_dvar", "bytes"),
    ],
)
def test_unsafe_object_local_data_is_rejected_before_schema_binding(
    tmp_path: Path,
    object_location: str,
    expected_kind: str,
) -> None:
    calls: list[int] = []
    critical_errors: list[Exception] = []

    def unsupported_frame() -> pd.DataFrame:
        if object_location == "local_ivar":
            return pd.DataFrame(
                {"value": [1.0, 2.0]},
                index=pd.Index(
                    [1, "1"],
                    dtype="object",
                    name="sample",
                ),
            )
        values = (
            pd.Series([b"a", b"b"], dtype="object")
            if object_location == "bytes_dvar"
            else pd.Series([1, "1"], dtype="object")
        )
        return pd.DataFrame(
            {
                "value": values
            },
            index=pd.Index([0, 1], name="sample"),
        )

    name = f"unsupported_mixed_object_{object_location}"
    manager = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or {"measurement": unsupported_frame()},
        ivars=_grid(1),
        meas_name=name,
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        critical_callback=critical_errors.append,
    )
    assert manager.run() is False
    assert calls == [1]
    assert len(critical_errors) == 1
    error_text = str(critical_errors[0]).lower()
    assert "object" in error_text and expected_kind in error_text
    assert not list(manager.output_dir.glob("chunk_*"))
    contract = json.loads(manager.contract_file.read_text(encoding="utf-8"))
    assert contract["result_schema"] is None
    assert contract["block_names"] is None
    assert contract["block_mode"] is None

    corrected_calls: list[int] = []
    corrected = SweepManager(
        measurement_func=lambda params: corrected_calls.append(params["point"])
        or _mapping_result(3.0),
        ivars=_grid(1),
        meas_name=name,
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    assert corrected.run() is True
    assert corrected_calls == [1]


def test_complex_local_index_is_rejected_but_complex_dvar_is_supported(
    tmp_path: Path,
) -> None:
    critical_errors: list[Exception] = []
    manager = SweepManager(
        measurement_func=lambda params: {
            "measurement": pd.DataFrame(
                {"value": [1.0]},
                index=pd.Index(
                    [1.0 + 2.0j],
                    dtype="complex128",
                    name="sample",
                ),
            )
        },
        ivars=_grid(1),
        meas_name="complex_local_index",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        critical_callback=critical_errors.append,
    )
    assert manager.run() is False
    assert len(critical_errors) == 1
    assert "complex" in str(critical_errors[0]).lower()
    assert not list(manager.output_dir.glob("chunk_*"))
    contract = json.loads(manager.contract_file.read_text(encoding="utf-8"))
    assert contract["result_schema"] is None
    assert contract["block_names"] is None
    assert contract["block_mode"] is None

    calls: list[int] = []
    corrected = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or {
            "measurement": pd.DataFrame(
                {"response": pd.Series([1.0 + 2.0j], dtype="complex128")},
                index=pd.Index([0], name="sample"),
            )
        },
        ivars=_grid(1),
        meas_name="complex_local_index",
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    assert corrected.run() is True
    assert calls == [1]
    result = corrected.get_results()
    assert isinstance(result, dict)
    assert result["measurement"]["response"].tolist() == [1.0 + 2.0j]

    target = tmp_path / "complex_dvar_allowed.h5"
    corrected.merge(target)
    GenericSweepDataset(target).validate_storage(deep=True)


def test_deep_table_validation_detects_duplicate_across_100k_chunks(
    tmp_path: Path,
) -> None:
    manager = SweepManager(
        measurement_func=lambda params: _mapping_result(1.0),
        ivars=_grid(1),
        meas_name="streaming_duplicate_seed",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    target = tmp_path / "streaming_duplicate_validation.h5"
    manager.partial_merge(target)

    first_samples = list(range(100_000))
    second_samples = [0, *range(100_001, 200_000)]
    assert len(first_samples) == len(second_samples) == 100_000
    index = pd.MultiIndex.from_arrays(
        [
            [1] * 200_000,
            [*first_samples, *second_samples],
        ],
        names=["point", "sample"],
    )
    replacement = pd.DataFrame(
        {"value": [1.0] * 200_000},
        index=index,
    )
    assert replacement.iloc[:100_000].index.is_unique
    assert replacement.iloc[100_000:].index.is_unique
    assert not replacement.index.is_unique

    with pd.HDFStore(target, mode="a") as store:
        store.remove("measurement")
        store.put(
            "measurement",
            replacement,
            format="table",
            data_columns=list(replacement.index.names),
        )

    dataset = GenericSweepDataset(target)
    with pytest.raises(ValueError, match="across table chunks"):
        dataset.validate_storage(deep=True)


def test_deep_table_validation_detects_missing_duplicate_across_chunks(
    tmp_path: Path,
) -> None:
    def measurement(_params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {
            "measurement": pd.DataFrame(
                {"value": [1.0]},
                index=pd.Index([0.0], name="sample"),
            )
        }

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1),
        meas_name="streaming_missing_duplicate_seed",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    target = tmp_path / "streaming_missing_duplicate_validation.h5"
    manager.partial_merge(target)

    first_samples = [float("nan"), *map(float, range(1, 100_000))]
    second_samples = [
        float("nan"),
        *map(float, range(100_001, 200_000)),
    ]
    index = pd.MultiIndex.from_arrays(
        [
            [1] * 200_000,
            [*first_samples, *second_samples],
        ],
        names=["point", "sample"],
    )
    replacement = pd.DataFrame(
        {"value": [1.0] * 200_000},
        index=index,
    )
    assert replacement.iloc[:100_000].index.is_unique
    assert replacement.iloc[100_000:].index.is_unique
    assert not replacement.index.is_unique

    with pd.HDFStore(target, mode="a") as store:
        store.remove("measurement")
        store.put(
            "measurement",
            replacement,
            format="table",
            data_columns=list(replacement.index.names),
        )

    dataset = GenericSweepDataset(target)
    with pytest.raises(ValueError, match="across table chunks"):
        dataset.validate_storage(deep=True)


@pytest.mark.skipif(os.name != "nt", reason="Windows path identity only")
def test_windows_merge_target_lock_is_case_insensitive(tmp_path: Path) -> None:
    upper = SweepManager._merge_target_lock(tmp_path / "Archive.H5")
    lower = SweepManager._merge_target_lock(tmp_path / "archive.h5")
    assert upper.path == lower.path


@pytest.mark.skipif(os.name != "nt", reason="Windows path identity only")
def test_windows_run_lock_is_case_insensitive_for_measurement_name(
    tmp_path: Path,
) -> None:
    first = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=_grid(1),
        meas_name="CaseVariantRun",
        output_root=tmp_path,
        resume=False,
    )
    first_lock_path = first._run_lock.path
    first.close()

    second = SweepManager(
        measurement_func=lambda params: _mapping_result(),
        ivars=_grid(1),
        meas_name="casevariantrun",
        output_root=tmp_path,
        resume=False,
    )
    second_lock_path = second._run_lock.path
    second.close()

    assert first_lock_path == second_lock_path


def test_resume_restarts_known_torn_write_only_directory_but_rejects_unknown(
    tmp_path: Path,
) -> None:
    known_name = "known_torn_write_remnants"
    known_dir = tmp_path / known_name
    known_dir.mkdir()
    torn_names = [
        "sweep_contract.json.tmp",
        "run_manifest.json.tmp",
        "simlog.tsv.tmp",
        "chunk_0.tmp.h5",
    ]
    for index, filename in enumerate(torn_names):
        (known_dir / filename).write_bytes(f"torn-{index}".encode("ascii"))

    calls: list[int] = []
    restarted = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or _mapping_result(5.0),
        ivars=_grid(1),
        meas_name=known_name,
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    assert all(not (known_dir / filename).exists() for filename in torn_names)
    assert restarted.run() is True
    assert calls == [1]
    assert restarted.contract_file.is_file()
    assert restarted.manifest_file.is_file()
    assert list(restarted.output_dir.glob("chunk_*.h5"))

    unknown_name = "unknown_resume_artifact"
    unknown_dir = tmp_path / unknown_name
    unknown_dir.mkdir()
    sentinel = unknown_dir / "operator-notes.txt"
    sentinel.write_bytes(b"must remain byte-exact")
    before = _snapshot_tree(unknown_dir)
    unknown_calls: list[int] = []

    with pytest.raises(RuntimeError, match="unknown files|valid.*artifacts"):
        SweepManager(
            measurement_func=lambda params: unknown_calls.append(
                params["point"]
            )
            or _mapping_result(),
            ivars=_grid(1),
            meas_name=unknown_name,
            output_root=tmp_path,
            resume=True,
        )

    assert unknown_calls == []
    assert _snapshot_tree(unknown_dir) == before


def test_uint64_global_sweep_level_is_rejected_before_callback_or_output(
    tmp_path: Path,
) -> None:
    ivars = pd.MultiIndex.from_arrays(
        [pd.Index([1, 2], dtype="uint64")],
        names=["point"],
    )
    calls: list[int] = []

    with pytest.raises(TypeError, match="uint64|checked int64"):
        SweepManager(
            measurement_func=lambda params: calls.append(params["point"])
            or _mapping_result(),
            ivars=ivars,
            meas_name="uint64_global_grid",
            output_root=tmp_path,
            resume=False,
        )

    assert calls == []
    assert not (tmp_path / "uint64_global_grid").exists()


def test_uint64_local_index_is_rejected_while_uint32_indices_and_dvar_work(
    tmp_path: Path,
) -> None:
    ivars = pd.MultiIndex.from_arrays(
        [pd.Index([1], dtype="uint32")],
        names=["point"],
    )
    critical_errors: list[Exception] = []
    manager = SweepManager(
        measurement_func=lambda params: {
                "measurement": pd.DataFrame(
                    {"value": [1.0]},
                    index=pd.Index([0], dtype="uint64", name="sample"),
            )
        },
        ivars=ivars,
        meas_name="uint64_local_index",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        critical_callback=critical_errors.append,
    )
    assert manager.run() is False
    assert len(critical_errors) == 1
    assert "uint64" in str(critical_errors[0]).lower()
    assert not list(manager.output_dir.glob("chunk_*"))
    contract = json.loads(manager.contract_file.read_text(encoding="utf-8"))
    assert contract["result_schema"] is None
    assert contract["block_names"] is None
    assert contract["block_mode"] is None

    calls: list[int] = []
    corrected = SweepManager(
        measurement_func=lambda params: calls.append(params["point"])
        or {
            "measurement": pd.DataFrame(
                {
                    "counter": pd.Series(
                        [2**63 + 5],
                        dtype="uint64",
                    )
                },
                index=pd.Index([0], dtype="uint32", name="sample"),
            )
        },
        ivars=ivars,
        meas_name="uint64_local_index",
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    )
    assert corrected.run() is True
    assert calls == [1]
    result = corrected.get_results()
    assert isinstance(result, dict)
    frame = result["measurement"]
    assert str(frame.index.get_level_values("point").dtype) == "uint32"
    assert str(frame.index.get_level_values("sample").dtype) == "uint32"
    assert str(frame["counter"].dtype) == "uint64"
    assert frame["counter"].tolist() == [2**63 + 5]

    target = tmp_path / "uint32_indices_uint64_dvar.h5"
    corrected.merge(target)
    dataset = GenericSweepDataset(target)
    stored = dataset.get_block("measurement")
    assert str(stored.index.get_level_values("point").dtype) == "uint32"
    assert str(stored.index.get_level_values("sample").dtype) == "uint32"
    assert str(stored["counter"].dtype) == "uint64"
    dataset.validate_storage(deep=True)


def test_partial_merge_prescans_string_width_and_keeps_sources_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_index = "index-" + "i" * 350
    long_value = "value-" + "v" * 400

    def measurement(params: dict[str, Any]):
        is_second = params["point"] == 2
        return {
            "measurement": pd.DataFrame(
                {
                    "text": [long_value if is_second else "short value"],
                },
                index=pd.Index(
                    [long_index if is_second else "short index"],
                    dtype="object",
                    name="label",
                ),
            )
        }

    manager = SweepManager(
        measurement_func=measurement,
        ivars=_grid(1, 2),
        meas_name="streaming_string_width",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )
    assert manager.run() is True
    chunk_files = sorted(manager.output_dir.glob("chunk_*.h5"))
    assert len(chunk_files) == 2

    target = tmp_path / "streaming_string_width.h5"
    manager.partial_merge(target)
    dataset = GenericSweepDataset(target)
    stored = dataset.get_block("measurement")
    assert long_index in stored.index.get_level_values("label")
    assert long_value in stored["text"].tolist()
    dataset.validate_storage(deep=True)

    source_bytes = {path.name: path.read_bytes() for path in chunk_files}
    failed_target = tmp_path / "streaming_string_width_failed.h5"

    def fail_metadata_commit(
        store: pd.HDFStore,
        *,
        drop_columns: Any = None,
    ) -> None:
        raise OSError("simulated merge metadata failure")

    monkeypatch.setattr(manager, "_write_metadata", fail_metadata_commit)
    with pytest.raises(OSError, match="simulated merge metadata failure"):
        manager.partial_merge(failed_target, remove_chunks=True)

    assert not failed_target.exists()
    assert {
        path.name: path.read_bytes()
        for path in sorted(manager.output_dir.glob("chunk_*.h5"))
    } == source_bytes
    manifest = json.loads(manager.manifest_file.read_text(encoding="utf-8"))
    assert manifest["state"] != "archived"


@pytest.mark.skipif(os.name != "nt", reason="Windows rename retry only")
def test_atomic_replace_retries_transient_windows_scanner_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pynst.sweep_manager as sweep_module

    temporary = tmp_path / "payload.tmp"
    target = tmp_path / "payload.json"
    temporary.write_bytes(b"durable payload")
    real_replace = os.replace
    calls = 0

    def transient_once(source: Any, destination: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError(
                13,
                "simulated scanner lock",
                str(destination),
                5,
            )
        real_replace(source, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(sweep_module.os, "replace", transient_once)
        patcher.setattr(sweep_module.time, "sleep", lambda delay: None)
        sweep_module._replace_file(temporary, target)

    assert calls == 2
    assert target.read_bytes() == b"durable payload"
    assert not temporary.exists()
