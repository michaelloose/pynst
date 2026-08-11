"""Storage validation tests for generic merged sweep datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pynst.data_model import BlockMetadata, SweepMetadata
from pynst.dataset import GenericSweepDataset


def _block_metadata(
    name: str,
    *,
    required: bool = True,
) -> BlockMetadata:
    return BlockMetadata(
        hdf_key=name,
        ivars=["point"],
        sweep_ivars=["point"],
        local_ivars=[],
        dvars=["value"],
        required=required,
    )


def _write_dataset(
    path: Path,
    *,
    frames: dict[str, pd.DataFrame],
    blocks: dict[str, BlockMetadata],
    schema_version: int = 2,
) -> None:
    metadata = SweepMetadata(
        measurement_name=path.stem,
        nested_sweep_levels=["point"],
        blocks=blocks,
        schema_version=schema_version,
    )
    with pd.HDFStore(path, mode="w") as store:
        for name, frame in frames.items():
            store.put(name, frame, format="table")
        store.put(
            "/__metadata__/config",
            pd.DataFrame({"json": [json.dumps(metadata.to_dict())]}),
            format="fixed",
        )


def _frame(
    points: list[int] | None = None,
    values: list[float] | None = None,
) -> pd.DataFrame:
    points = [0, 1] if points is None else points
    values = [1.0, 2.0] if values is None else values
    return pd.DataFrame(
        {"value": values},
        index=pd.Index(points, name="point"),
    )


def test_dataset_path_must_reference_a_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.h5"
    with pytest.raises(FileNotFoundError):
        GenericSweepDataset(missing)

    with pytest.raises(ValueError, match="must reference a file"):
        GenericSweepDataset(tmp_path)


def test_nested_hdf_block_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nested.h5"
    with pd.HDFStore(path, mode="w") as store:
        store.put("/group/block", _frame(), format="table")

    with pytest.raises(ValueError, match="must not contain"):
        GenericSweepDataset(path, validate=False)


def test_block_lookup_rejects_nested_names(tmp_path: Path) -> None:
    path = tmp_path / "valid.h5"
    _write_dataset(
        path,
        frames={"measurement": _frame()},
        blocks={"measurement": _block_metadata("measurement")},
    )
    dataset = GenericSweepDataset(path)

    with pytest.raises(ValueError, match="must not contain"):
        dataset.get_block("measurement/subblock")


def test_nested_metadata_block_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nested_metadata.h5"
    _write_dataset(
        path,
        frames={"measurement": _frame()},
        blocks={
            "measurement": _block_metadata("measurement"),
            "group/optional": _block_metadata(
                "group/optional",
                required=False,
            ),
        },
    )

    with pytest.raises(ValueError, match="must not contain"):
        GenericSweepDataset(path)


def test_deep_validation_checks_the_complete_index(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.h5"
    _write_dataset(
        path,
        frames={
            "measurement": _frame(
                points=[0, 1, 1],
                values=[1.0, 2.0, 3.0],
            )
        },
        blocks={"measurement": _block_metadata("measurement")},
    )
    dataset = GenericSweepDataset(path)

    with pytest.raises(ValueError, match="duplicate entries"):
        dataset.validate_storage(deep=True)


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (
            pd.DataFrame(
                {"value": [1.0]},
                index=pd.Index([0], name="wrong_index"),
            ),
            "index names differ",
        ),
        (
            pd.DataFrame(
                {"wrong_value": [1.0]},
                index=pd.Index([0], name="point"),
            ),
            "columns differ",
        ),
    ],
)
def test_deep_validation_checks_index_names_and_columns(
    tmp_path: Path,
    frame: pd.DataFrame,
    error: str,
) -> None:
    path = tmp_path / "structure.h5"
    _write_dataset(
        path,
        frames={"measurement": frame},
        blocks={"measurement": _block_metadata("measurement")},
    )
    dataset = GenericSweepDataset(path)

    with pytest.raises(ValueError, match=error):
        dataset.validate_storage(deep=True)


def test_versioned_dataset_rejects_physical_block_without_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unexpected.h5"
    _write_dataset(
        path,
        frames={
            "measurement": _frame(),
            "unexpected": _frame(),
        },
        blocks={"measurement": _block_metadata("measurement")},
    )
    dataset = GenericSweepDataset(path)

    # The shallow constructor check remains permissive for cheap loading.
    with pytest.raises(ValueError, match="no metadata entries"):
        dataset.validate_storage(deep=True)


def test_required_metadata_block_is_still_enforced(tmp_path: Path) -> None:
    path = tmp_path / "missing_required.h5"
    _write_dataset(
        path,
        frames={"measurement": _frame()},
        blocks={
            "measurement": _block_metadata("measurement"),
            "required_aux": _block_metadata("required_aux"),
        },
    )

    with pytest.raises(ValueError, match="Required HDF blocks are missing"):
        GenericSweepDataset(path)


def test_missing_optional_metadata_block_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "missing_optional.h5"
    _write_dataset(
        path,
        frames={"measurement": _frame()},
        blocks={
            "measurement": _block_metadata("measurement"),
            "optional_aux": _block_metadata(
                "optional_aux",
                required=False,
            ),
        },
    )

    dataset = GenericSweepDataset(path)
    dataset.validate_storage(deep=True)


def test_legacy_file_without_config_remains_deep_validatable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.h5"
    with pd.HDFStore(path, mode="w") as store:
        store.put("measurement", _frame(), format="table")

    dataset = GenericSweepDataset(path)

    assert dataset.metadata.schema_version == 0
    assert dataset.metadata.extra["legacy_file"] is True
    dataset.validate_storage(deep=True)


def test_deep_validation_rejects_materialised_empty_table_block(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty_table.h5"
    _write_dataset(
        path,
        frames={"measurement": _frame()},
        blocks={"measurement": _block_metadata("measurement")},
    )
    with pd.HDFStore(path, mode="a") as store:
        assert store.remove("measurement", start=0, stop=2) == 2
        assert store.get_storer("measurement").nrows == 0

    dataset = GenericSweepDataset(path)
    with pytest.raises(ValueError, match="block is empty"):
        dataset.validate_storage(deep=True)
